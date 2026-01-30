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
flags.DEFINE_integer("eval_checkpoint_step", 63000, "Checkpoint step to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 1, "Number of evaluation episodes.")
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

jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "/home/terry/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
print(f"JAX cache dir: {jax.config.values['jax_compilation_cache_dir']}")
print(f"JAX cache enabled: {jax.config.values.get('jax_enable_compilation_cache', 'Not set')}")
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

def wait_for_file(path: str, poll_s: float = 0.05, remove_first: bool = True):
    p = Path(path)
    # don't accidentally auto-start due to a stale file
    if remove_first and p.exists():
        p.unlink()
    print(f"[eval] Ready. Waiting for: {path}", flush=True)
    while not p.exists():
        time.sleep(poll_s)
    print("[eval] Start signal received.", flush=True)
    

def restore_agent_checkpoint(agent):
    assert FLAGS.checkpoint_path is not None, "--checkpoint_path is required"
    assert FLAGS.eval_checkpoint_step > 0, "--eval_checkpoint_step must be > 0"
    assert FLAGS.eval_n_trajs > 0, "--eval_n_trajs must be > 0"

    t0 = time.time()
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)
    print_green(f"[eval] checkpoint restored in {time.time()-t0:.3f}s")
    return agent



def to_policy_obs(obs):
    """Convert env obs to the policy's expected obs dict.

    Policy (SAC encoder) expects:
      - image keys (e.g., 'wrist_1') at the TOP level
      - optional proprio under key 'state' as a 1D float array (not a dict)

    FrankaEnv returns:
      {'images': {cam: HxWx3 uint8}, 'state': {tcp_pose,...}}
    This function flattens it to:
      {cam: HxWx3 uint8, 'state': (20,) float32}
    """
    if not (isinstance(obs, dict) and "images" in obs and isinstance(obs["images"], dict)):
        return obs

    out = dict(obs["images"])

    # Flatten state dict -> vector (matches FrankaEnv.observation_space)
    st = obs.get("state", None)
    if isinstance(st, dict):
        parts = []
        for k in ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"]:
            if k not in st:
                continue
            v = np.asarray(st[k], dtype=np.float32).reshape(-1)
            parts.append(v)
        if parts:
            out["state"] = np.concatenate(parts, axis=0).astype(np.float32)
    elif st is not None:
        out["state"] = np.asarray(st, dtype=np.float32)

    return out


def warmup_policy(agent, rng, obs):
    """Trigger JAX compilation before we start moving the robot.

    Option B: warm up using a *real* observation pytree (same structure/dtypes as eval loop),
    without calling env.reset()/env.step().
    """
    t0 = time.time()
    print_green("[eval] warming up policy (JIT compile)")

    # Match the evaluation loop behavior (split rng, then call sample_actions).
    rng, key = jax.random.split(rng)
    obs = to_policy_obs(obs)

    actions = agent.sample_actions(
        observations=jax.device_put(to_policy_obs(obs)),
        argmax=True,
        seed=key,
    )

    # Block until compilation/execution completes.
    try:
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            actions,
        )
    except Exception:
        # Fallback: block on the first leaf if tree_map fails for any reason.
        leaves, _ = jax.tree_util.tree_flatten(actions)
        if leaves and hasattr(leaves[0], "block_until_ready"):
            leaves[0].block_until_ready()

    print_green(f"[eval] warmup finished in {time.time()-t0:.3f}s")
    return rng


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

    def move_gripper(self, pos: int):
        pos = int(np.clip(pos, 0, 1000))
        self._post("move_gripper", json={"gripper_pos": pos})

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

    home = np.array([0.13387135, 0.55630949, 0.16354294, np.pi, 0, 0], dtype=np.float64)

    robot.gripper_open()
    time.sleep(0.1)

    st = robot.get_state()
    curr = np.asarray(st["pose"], dtype=np.float64).reshape(7)
    target_pose = curr.copy()
    lift_pose = target_pose.copy()
    lift_pose[2] += 0.04

    robot.moveL(lift_pose, timeout=1.0)
    robot.move_gripper(10)

    time.sleep(0.1)
    robot.moveL(target_pose, timeout=1.0)
    
    robot.moveL(home, timeout=1.0)
    robot.gripper_open()

def evaluate(agent, env, rng):

    success_counter = 0.0
    times = []

    for ep in range(int(FLAGS.eval_n_trajs)):
        t_imp = time.time()
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
                argmax=True,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))

            print_green(f"[timing] warm up time: {time.time() - t_imp:.3f}s")
            obs, reward, done, truncated_step, info = env.step(actions)
            last_info = info if isinstance(info, dict) else {}
            truncated = truncated or bool(truncated_step)
            step_in_ep += 1

        succeed = bool(last_info.get("succeed")) if "succeed" in last_info else False
        success_counter += float(succeed)
        if succeed:
            times.append(time.time() - t0)

        print_green(f"[{ep+1}/{FLAGS.eval_n_trajs}] success={succeed} steps={step_in_ep}")

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
    # Clear any stale start signal file once at startup.
    start_file = "/tmp/start_eval"
    p = Path(start_file)
    if p.exists():
        p.unlink()

    print_green("stop impedance for host controller")
    robot.stop_imp()

    # Preload checkpoint + JIT compile while the host controller is running.
    agent = restore_agent_checkpoint(agent)
    # Option B warmup: use a real observation (no reset/step), so JAX compiles the exact eval path.
    warm_obs = None
    try:
        # base_env here is env.unwrapped (created above); it should expose _update_currpos/_get_obs in FrankaEnv.
        if hasattr(base_env, "_update_currpos"):
            base_env._update_currpos()
        if hasattr(base_env, "_get_obs"):
            warm_obs = base_env._get_obs()
    except Exception as e:
        print(f"[warn] warmup obs via base_env._get_obs failed: {e}")

    if warm_obs is None:
        # Fallback (still compiles something shape-correct)
        warm_obs = env.observation_space.sample()
        print("[warn] using observation_space.sample() for warmup (fallback).")

    sampling_rng = warmup_policy(agent, sampling_rng, warm_obs)
    wait_for_file(start_file, remove_first=False)
    
    print_green("start impedance and evaluate")
    t_imp = time.time()
    robot.start_imp()
    print_green(f"[timing] time for start_imp: {time.time() - t_imp:.3f}s")
    evaluate(agent, env, sampling_rng)

    scripted_post_episode(robot)

if __name__ == "__main__":
    app.run(main)
