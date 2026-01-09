#!/usr/bin/env python3

import os
import time
import datetime
from pathlib import Path
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
flags.DEFINE_float("script_default_duration", 5.0, "Default seconds for moveL interpolation.")

flags.DEFINE_boolean("do_pickup", False, "Run pickup scripted routine before policy rollout.")
flags.DEFINE_boolean("do_release", False, "Open gripper after the episode ends.")
flags.DEFINE_boolean("do_joint_reset_after", False, "Call joint reset after each episode ends.")

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

def quat_angle(q1, q2):
    # q = [x,y,z,w]
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    d = abs(float(np.dot(q1, q2)))
    d = max(-1.0, min(1.0, d))
    return 2.0 * np.arccos(d)  # rad

def wait_for_file(path: str, poll_s: float = 0.05):
    p = Path(path)
    # don't accidentally auto-start due to a stale file
    if p.exists():
        p.unlink()
    print(f"[eval] Ready. Waiting for: {path}", flush=True)
    while not p.exists():
        time.sleep(poll_s)
    print("[eval] Start signal received.", flush=True)
    
@dataclass
class FlaskRobotClient:
    server_url: str
    hz: float = 10.0
    timeout_s: float = 10.0

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

    def start_imp(self):
        self._post("startimp")

    def stop_imp(self):
        self._post("stopimp")

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

def scripted_post_episode(robot: FlaskRobotClient):
    """Example post-episode routine: (optional) move -> open -> (optional) move -> joint reset."""
    pre_release = _parse_floats_csv(FLAGS.post_release_pose)
    post_safe = _parse_floats_csv(FLAGS.post_safe_pose)

    if pre_release is not None:
        print_green("[script] post: moveL pre-release")
        robot.moveL(pre_release, timeout=FLAGS.script_default_duration)

    if FLAGS.do_release:
        print_green("[script] post: open gripper")
        robot.gripper_open()
        time.sleep(0.2)

    if post_safe is not None:
        print_green("[script] post: moveL safe pose")
        robot.moveL(post_safe, timeout=FLAGS.script_default_duration)

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

    robot = FlaskRobotClient(server_url=server_url, hz=hz)
    print_green(f"Flask server: {robot.server_url}")
    print_green("stop impedance for host controller")
    robot.stop_imp()

    wait_for_file("/tmp/start_eval")
    
    print_green("start impedance and evaluate")
    robot.start_imp()
    evaluate(agent, env, sampling_rng, robot)


if __name__ == "__main__":
    app.run(main)
