#!/usr/bin/env python3
"""Actor-only evaluation script with *scriptable* robot/gripper utilities.

This is a companion to eval_record_minimal.py:
- eval_record_minimal.py: evaluate + RECORD full rollouts to pkl.
- this script: evaluate WITHOUT recording, but lets you run scripted motions
  (pickup, release, joint reset, etc.) via small reusable util functions.

Why this works outside the docker:
- In HIL-SERL's FrankaEnv, robot/gripper are controlled via a Flask server.
  The env calls HTTP endpoints like /pose, /open_gripper, /close_gripper,
  /jointreset, /getstate.
- This script uses the same HTTP endpoints for scripted segments, and uses
  env.step() for the policy rollout.
"""

import os
import time
import datetime
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import requests
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)

FLAGS = flags.FLAGS

# --- core eval flags ---
flags.DEFINE_string("exp_name", None, "Experiment name (key in CONFIG_MAPPING).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint directory.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Checkpoint step to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of evaluation episodes.")
flags.DEFINE_boolean("save_video", False, "Enable env-side video saving if supported.")
flags.DEFINE_integer("eval_max_steps_per_ep", 0, "Optional cap (0 = until env done).")

# --- scripted utilities flags ---
flags.DEFINE_string(
    "server_url_override",
    "",
    "Optional override for Flask server URL (e.g., http://127.0.0.1:5000/). "
    "If empty, uses env.unwrapped.url.",
)
flags.DEFINE_float("script_hz", 10.0, "Hz for scripted moveL interpolation.")
flags.DEFINE_float("script_default_timeout", 1.0, "Default seconds for moveL interpolation.")

flags.DEFINE_boolean("do_pickup", False, "Run pickup scripted routine before policy rollout.")
flags.DEFINE_boolean("do_release", True, "Open gripper after the episode ends.")
flags.DEFINE_boolean("do_joint_reset_after", True, "Call joint reset after each episode ends.")

# Poses are passed as comma-separated floats.
# - moveL accepts either 6D (x,y,z,roll,pitch,yaw) in radians or 7D (x,y,z,qx,qy,qz,qw).
flags.DEFINE_string("pickup_pre_pose", "", "Optional 6D/7D pose string for pre-grasp.")
flags.DEFINE_string("pickup_grasp_pose", "", "Optional 6D/7D pose string for grasp pose.")
flags.DEFINE_string("pickup_lift_pose", "", "Optional 6D/7D pose string for lift/retreat pose.")

flags.DEFINE_string("post_release_pose", "", "Optional 6D/7D pose to move to before opening gripper.")
flags.DEFINE_string("post_safe_pose", "", "Optional 6D/7D pose to move to after opening gripper.")


# Keep consistent with the training/eval pattern: replicate across local devices.
devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x: str):
    print("\033[92m {}\033[00m".format(x))


def _parse_floats_csv(s: str) -> Optional[np.ndarray]:
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    arr = np.array([float(p) for p in parts], dtype=np.float64)
    if arr.size not in (6, 7):
        raise ValueError(f"Expected 6 or 7 floats, got {arr.size}: {s}")
    return arr


def _rpy_to_quat_xyz_w(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (rad) to quaternion [x, y, z, w]."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _pose6_or_7_to_pose7(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.size == 7:
        return p
    xyz = p[:3]
    quat = _rpy_to_quat_xyz_w(p[3:])
    return np.concatenate([xyz, quat], axis=0)


@dataclass
class FlaskRobotClient:
    server_url: str
    hz: float = 10.0
    timeout_s: float = 5.0

    def _post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.server_url.rstrip("/") + "/" + endpoint.lstrip("/")
        r = requests.post(url, json=json, timeout=self.timeout_s)
        # Some endpoints may not return JSON; handle best-effort.
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}

    def clear_error(self):
        self._post("clearerr")

    def get_state(self) -> Dict[str, Any]:
        return self._post("getstate")

    def send_pose7(self, pose7: np.ndarray):
        pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
        self.clear_error()
        self._post("pose", json={"arr": pose7.tolist()})

    def gripper_open(self):
        self._post("open_gripper")

    def gripper_close(self):
        self._post("close_gripper")

    def joint_reset(self):
        self._post("jointreset")

    def moveL(self, goal_pose: np.ndarray, timeout: Optional[float] = None):
        """Linear-interp Cartesian motion by repeatedly calling /pose at hz.

        goal_pose: (6,) xyz+rpy(rad) OR (7,) xyz+quat(xyzw)
        """
        timeout = float(timeout) if timeout is not None else 1.0
        hz = float(self.hz) if self.hz > 0 else 10.0
        steps = max(1, int(round(timeout * hz)))

        st = self.get_state()
        curr = np.asarray(st["pose"], dtype=np.float64).reshape(7)
        goal = _pose6_or_7_to_pose7(goal_pose)

        path = np.linspace(curr, goal, steps)
        self.clear_error()
        for p in path:
            self.send_pose7(p)
            time.sleep(1.0 / hz)

    
def moveJ(self, q: Optional[Sequence[float]] = None, timeout: Optional[float] = None):
    """Joint-space move.

    - If q is None: calls /jointreset (built-in in your server).
    - If q is provided: calls /movej with {"q": [7 joints], "timeout_s": ...}.

    Requires the Flask server to expose POST /movej.
    """
    if q is None:
        self.joint_reset()
        time.sleep(0.5)
        return

    q_list = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
    if len(q_list) != 7:
        raise ValueError(f"moveJ expects 7 joints (rad), got {len(q_list)}")

    payload = {"q": q_list}
    if timeout is not None:
        payload["timeout_s"] = float(timeout)

    out = self._post("movej", json=payload)
    if not isinstance(out, dict):
        raise RuntimeError(f"/movej returned non-JSON: {out}")

    if out.get("success") is True:
        return

    raise RuntimeError(f"moveJ failed: {out}")

def scripted_pickup(robot: FlaskRobotClient):
    """Example pickup routine (poses must be provided via flags)."""
    pre = _parse_floats_csv(FLAGS.pickup_pre_pose)
    grasp = _parse_floats_csv(FLAGS.pickup_grasp_pose)
    lift = _parse_floats_csv(FLAGS.pickup_lift_pose)

    if pre is None or grasp is None:
        print_green("[script] pickup skipped (need --pickup_pre_pose and --pickup_grasp_pose).")
        return

    print_green("[script] pickup: open gripper")
    robot.gripper_open()
    time.sleep(0.2)

    print_green("[script] pickup: moveL pre-grasp")
    robot.moveL(pre, timeout=FLAGS.script_default_timeout)

    print_green("[script] pickup: moveL grasp")
    robot.moveL(grasp, timeout=FLAGS.script_default_timeout)

    print_green("[script] pickup: close gripper")
    robot.gripper_close()
    time.sleep(0.4)

    if lift is not None:
        print_green("[script] pickup: moveL lift/retreat")
        robot.moveL(lift, timeout=FLAGS.script_default_timeout)


def scripted_post_episode(robot: FlaskRobotClient):
    """Example post-episode routine: (optional) move -> open -> (optional) move -> joint reset."""
    pre_release = _parse_floats_csv(FLAGS.post_release_pose)
    post_safe = _parse_floats_csv(FLAGS.post_safe_pose)

    if pre_release is not None:
        print_green("[script] post: moveL pre-release")
        robot.moveL(pre_release, timeout=FLAGS.script_default_timeout)

    if FLAGS.do_release:
        print_green("[script] post: open gripper")
        robot.gripper_open()
        time.sleep(0.2)

    if post_safe is not None:
        print_green("[script] post: moveL safe pose")
        robot.moveL(post_safe, timeout=FLAGS.script_default_timeout)

    if FLAGS.do_joint_reset_after:
        print_green("[script] post: joint reset")
        robot.moveJ(None)
        time.sleep(0.5)


def evaluate(agent, env, rng, robot: FlaskRobotClient):
    assert FLAGS.checkpoint_path is not None, "--checkpoint_path is required"
    assert FLAGS.eval_checkpoint_step, "--eval_checkpoint_step must be > 0"
    assert FLAGS.eval_n_trajs > 0, "--eval_n_trajs must be > 0"

    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)

    success_counter = 0.0
    times = []

    for ep in range(int(FLAGS.eval_n_trajs)):
        obs, _ = env.reset()
        done = False
        truncated = False
        t0 = time.time()

        # Scripted "setup" (pickup) BEFORE the policy loop.
        if FLAGS.do_pickup:
            scripted_pickup(robot)
            # ensure the env sees fresh state if it caches (FrankaEnv queries /getstate in step anyway)

        step_in_ep = 0
        last_info: Dict[str, Any] = {}

        while not (done or truncated):
            if FLAGS.eval_max_steps_per_ep and step_in_ep >= FLAGS.eval_max_steps_per_ep:
                truncated = True
                break

            rng, key = jax.random.split(rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=False,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))

            obs, reward, done, truncated_step, info = env.step(actions)
            last_info = info if isinstance(info, dict) else {}
            truncated = truncated or bool(truncated_step)
            step_in_ep += 1

        succeed = bool(last_info.get("succeed")) if "succeed" in last_info else False
        success_counter += float(succeed)
        if succeed:
            times.append(time.time() - t0)

        print_green(f"[{ep+1}/{FLAGS.eval_n_trajs}] success={succeed} steps={step_in_ep}")

        # Scripted "teardown" AFTER the policy loop.
        scripted_post_episode(robot)

    print_green(f"success rate: {success_counter / float(FLAGS.eval_n_trajs)}")
    if len(times):
        print_green(f"avg time (success only): {float(np.mean(times)):.3f}s")
    else:
        print_green("avg time (success only): nan")


def main(_):
    assert FLAGS.exp_name is not None, "--exp_name is required"
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment name not found in CONFIG_MAPPING"
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Build env
    env = config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=True)
    env = RecordEpisodeStatistics(env)

    # Create robot client (same server as env)
    base_env = getattr(env, "unwrapped", env)
    server_url = FLAGS.server_url_override.strip() or getattr(base_env, "url", "http://127.0.0.1:5000/")
    hz = float(FLAGS.script_hz) if FLAGS.script_hz > 0 else float(getattr(base_env, "hz", 10.0))
    robot = FlaskRobotClient(server_url=server_url, hz=hz)

    # Build agent
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # Seed + put agent/rng on devices (replicated)
    rng = jax.random.PRNGKey(int(FLAGS.seed))
    rng, sampling_rng = jax.random.split(rng)

    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    print_green("starting evaluation (scripted + policy)")
    print_green(f"Flask server: {robot.server_url}")
    evaluate(agent, env, sampling_rng, robot)


if __name__ == "__main__":
    app.run(main)
