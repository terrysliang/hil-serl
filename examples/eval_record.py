#!/usr/bin/env python3
"""
HIL-SERL / SERL evaluation + rollout recording script (actor-only).

- Loads an experiment config from experiments.mappings.CONFIG_MAPPING[exp_name]
- Builds the environment
- Builds the SAC(-hybrid) pixel agent matching the config.setup_mode
- Restores a checkpoint
- Runs evaluation rollouts and records each episode as a pickle file

This script intentionally removes learner/trainer/replay-buffer code.
"""

import os
import time
import datetime
import pickle as pkl


import sys
import select
import termios
import tty
import atexit
import requests
import jax
import jax.numpy as jnp
import numpy as np
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

flags.DEFINE_string("exp_name", None, "Experiment name (key in CONFIG_MAPPING).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint directory.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Checkpoint step to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of evaluation episodes.")
flags.DEFINE_boolean("save_video", False, "Enable env-side video saving if supported.")

flags.DEFINE_string(
    "eval_record_dir",
    None,
    "Directory to save evaluation rollouts. Default: <checkpoint_path>/eval_rollouts_step_<step>",
)
flags.DEFINE_string(
    "eval_task",
    "",
    "Optional task description/prompt to store with each recorded episode.",
)
flags.DEFINE_list(
    "eval_camera_keys",
    ["wrist_1", "wrist_2"],
    "Camera keys to keep under obs['images']. Use [] to keep all.",
)
flags.DEFINE_list(
    "record_camera_keys",
    ["wrist_1_full", "wrist_2_full", "side_1"],
    "Camera keys to keep under obs['images']. Use [] to keep all.",
)
flags.DEFINE_boolean(
    "eval_record_infos",
    False,
    "Whether to record per-step `info` dict (can be large / non-serializable).",
)
flags.DEFINE_integer(
    "eval_max_steps_per_ep",
    0,
    "Optional cap on steps per evaluation episode (0 = until termination).",
)


jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "/home/terry/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
print(f"JAX cache dir: {jax.config.values['jax_compilation_cache_dir']}")
print(f"JAX cache enabled: {jax.config.values.get('jax_enable_compilation_cache', 'Not set')}")

# JAX sharding (keep consistent with training code path)
devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x: str):
    print("\033[92m {}\033[00m".format(x))


# ---------------------------
# Interactive keyboard controls (TTY only)
# ---------------------------

def _is_tty():
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


class KeyReader:
    # Non-blocking single-key reader for terminal (Linux).
    def __init__(self):
        self.enabled = _is_tty()
        self._fd = None
        self._old = None

    def start(self):
        if not self.enabled:
            return
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.stop)

    def stop(self):
        if not self.enabled:
            return
        try:
            if self._fd is not None and self._old is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass
        self._fd = None
        self._old = None

    def get_key(self, timeout_s: float = 0.0):
        # Return a single character if available, else None.
        if not self.enabled:
            return None
        try:
            r, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if r:
                return sys.stdin.read(1)
        except Exception:
            return None
        return None


def _get_base_env(env):
    base = getattr(env, "unwrapped", env)
    while hasattr(base, "env") and hasattr(base.env, "unwrapped") and base.env is not base:
        base = base.env.unwrapped
    return base


def _robot_open(base_env):
    if hasattr(base_env, "url"):
        requests.post(base_env.url + "open_gripper")
        return True
    return False


def _robot_close(base_env):
    if hasattr(base_env, "url"):
        requests.post(base_env.url + "close_gripper")
        return True
    return False


def _print_controls():
    print_green(
        """=== EVAL RECORD CONTROLS ===
   o/c : open/close gripper immediately
   p   : pause/resume env stepping (safe while adjusting peg)
   r   : drop current episode buffer and reset env immediately
   q   : quit evaluation loop
   (after episode ends) k = keep+save episode, x = drop+redo episode
"""
    )

def _to_numpy_pytree(x):
    """Convert nested structures (dict/list/tuple) of arrays/scalars to numpy arrays."""
    return jax.tree_map(lambda y: np.asarray(y), x)


def _filter_obs_images(obs_np, camera_keys):
    if not camera_keys:
        return obs_np
    if not isinstance(obs_np, dict):
        return obs_np

    # Case A: cameras under obs["images"]
    if "images" in obs_np and isinstance(obs_np["images"], dict):
        imgs = obs_np["images"]
        missing = [k for k in camera_keys if k not in imgs]
        if missing:
            print_green(f"[eval_record] warning: missing cameras in obs['images']: {missing}")
        out = dict(obs_np)
        out["images"] = {k: imgs[k] for k in camera_keys if k in imgs}
        return out

    # Case B: cameras are top-level keys (obs["wrist_1"], obs["wrist_1_full"], ...)
    out = dict(obs_np)
    for k, v in list(out.items()):
        if k in camera_keys:
            continue
        arr = np.asarray(v)
        if isinstance(arr, np.ndarray) and arr.ndim >= 3:
            out.pop(k, None)
    return out


def _obs_for_policy(obs, policy_image_keys):
    """Return an obs that keeps only the cameras used by the policy.
    Leaves other (non-image) fields intact.
    """
    if not policy_image_keys or not isinstance(obs, dict):
        return obs

    # Case A: cameras are under obs["images"]
    if "images" in obs and isinstance(obs["images"], dict):
        out = dict(obs)
        imgs = obs["images"]
        out["images"] = {k: imgs[k] for k in policy_image_keys if k in imgs}
        return out

    # Case B: cameras are top-level keys (e.g., obs["wrist_1"])
    # Heuristic: drop extra image-like arrays (ndim>=3) that are not in policy_image_keys.
    out = dict(obs)
    for k, v in list(out.items()):
        if k in policy_image_keys:
            continue
        ndim = getattr(v, "ndim", None)
        if ndim is not None and ndim >= 3:
            out.pop(k, None)
    return out


def _env_rate_info(env):
    """Best-effort extract control rate info from unwrapped env."""
    base = getattr(env, "unwrapped", env)
    hz = float(getattr(base, "hz", 10.0))
    dt = 1.0 / hz if hz > 0 else 0.1
    action_scale = None
    if hasattr(base, "action_scale"):
        try:
            action_scale = np.asarray(getattr(base, "action_scale")).tolist()
        except Exception:
            action_scale = None
    return hz, dt, action_scale


def eval_and_record(agent, env, rng, policy_image_keys):
    assert FLAGS.checkpoint_path is not None, "--checkpoint_path is required"
    assert FLAGS.eval_checkpoint_step, "--eval_checkpoint_step must be > 0"
    assert FLAGS.eval_n_trajs > 0, "--eval_n_trajs must be > 0"

    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)

    if FLAGS.eval_record_dir is not None:
        record_dir = FLAGS.eval_record_dir
    else:
        record_dir = os.path.join(
            os.path.abspath(FLAGS.checkpoint_path),
            f"eval_rollouts_step_{FLAGS.eval_checkpoint_step}",
        )
    os.makedirs(record_dir, exist_ok=True)

    hz, dt, action_scale = _env_rate_info(env)

    meta = dict(
        exp_name=FLAGS.exp_name,
        seed=int(FLAGS.seed),
        checkpoint_path=os.path.abspath(FLAGS.checkpoint_path),
        eval_checkpoint_step=int(FLAGS.eval_checkpoint_step),
        eval_n_trajs=int(FLAGS.eval_n_trajs),
        control_hz=hz,
        dt=dt,
        action_semantics="delta_pose",
        action_scale=action_scale,
        camera_keys=list(FLAGS.record_camera_keys),
        task=FLAGS.eval_task,
        recorded_at=datetime.datetime.now().isoformat(),
    )

    success_counter = 0.0
    success_times = []

    
    # Interactive controls
    keyr = KeyReader()
    keyr.start()
    _print_controls()

    episode = 0
    pending_reset = None  # (obs, info) if we already reset manually
    while episode < int(FLAGS.eval_n_trajs):
        if pending_reset is None:
            obs, _ = env.reset()
        else:
            obs, _ = pending_reset
            pending_reset = None
        done = False
        truncated = False
        start_time = time.time()

        ep = dict(
            meta=meta,
            episode_index=int(episode),
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            masks=[],
            truncated=[],
            next_observations=[],
        )
        if FLAGS.eval_record_infos:
            ep["infos"] = []

        step_in_ep = 0
        last_info = {}

        base_env = _get_base_env(env)
        paused = False
        redo_episode = False
        quit_all = False

        while not (done or truncated):
            # Handle keys (non-blocking)
            ch = keyr.get_key(timeout_s=0.0)
            if ch:
                ch = ch.lower()
                if ch == "p":
                    paused = not paused
                    print_green(f"[eval_record] paused={paused}")
                elif ch == "o":
                    if _robot_open(base_env):
                        print_green("[eval_record] gripper_open sent")
                elif ch == "c":
                    if _robot_close(base_env):
                        print_green("[eval_record] gripper_close sent")
                elif ch == "r":
                    print_green("[eval_record] dropping current episode + resetting env")
                    redo_episode = True
                    break
                elif ch == "q":
                    print_green("[eval_record] quitting")
                    quit_all = True
                    break

            if quit_all:
                break

            if paused:
                time.sleep(0.05)
                continue

            if FLAGS.eval_max_steps_per_ep and step_in_ep >= FLAGS.eval_max_steps_per_ep:
                truncated = True
                break

            rng, key = jax.random.split(rng)

            policy_obs = _obs_for_policy(obs, policy_image_keys)

            actions = agent.sample_actions(
                observations=jax.device_put(policy_obs),
                argmax=True,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated_step, info = env.step(actions)
            last_info = info
            truncated = truncated or bool(truncated_step)

            obs_np = _filter_obs_images(_to_numpy_pytree(obs), FLAGS.record_camera_keys)
            next_obs_np = _filter_obs_images(_to_numpy_pytree(next_obs), FLAGS.record_camera_keys)

            info_to_store = info
            if isinstance(info_to_store, dict):
                info_to_store = dict(info_to_store)
                info_to_store.pop("left", None)
                info_to_store.pop("right", None)

            ep["observations"].append(obs_np)
            ep["actions"].append(np.asarray(actions))
            ep["rewards"].append(float(reward))
            ep["dones"].append(bool(done))
            ep["masks"].append(float(1.0 - float(done)))
            ep["truncated"].append(bool(truncated_step))
            ep["next_observations"].append(next_obs_np)
            if FLAGS.eval_record_infos:
                ep["infos"].append(info_to_store)

            obs = next_obs
            step_in_ep += 1

        if quit_all:
            break

        if redo_episode:
            try:
                pending_reset = env.reset()
            except Exception:
                pending_reset = None
            continue

        ep["final_observation"] = _filter_obs_images(_to_numpy_pytree(obs), FLAGS.record_camera_keys)
        ep["num_steps"] = int(step_in_ep)

        succeed = False
        if isinstance(last_info, dict) and "succeed" in last_info:
            succeed = bool(last_info.get("succeed"))
        elif len(ep["rewards"]) > 0:
            succeed = bool(ep["rewards"][-1])
        ep["success"] = succeed

        print_green(f"[episode ended] success={succeed} steps={step_in_ep}. Press k=keep, x=drop, q=quit.")
        decision = None
        while decision is None:
            ch = keyr.get_key(timeout_s=0.1)
            if ch:
                ch = ch.lower()
                if ch == "k":
                    decision = "keep"
                elif ch == "x":
                    decision = "drop"
                elif ch == "q":
                    decision = "quit"
                elif ch == "o":
                    if _robot_open(base_env):
                        print_green("[eval_record] gripper_open sent")
                elif ch == "c":
                    if _robot_close(base_env):
                        print_green("[eval_record] gripper_close sent")
                elif ch == "r":
                    decision = "drop"

        if decision == "quit":
            break

        if decision == "drop":
            print_green("[eval_record] dropped episode; redoing same index")
            try:
                pending_reset = env.reset()
            except Exception:
                pending_reset = None
            continue

        # decision == keep
        if succeed:
            success_times.append(time.time() - start_time)
        success_counter += float(succeed)

        out_path = os.path.join(record_dir, f"episode_{episode:04d}.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(ep, f, protocol=pkl.HIGHEST_PROTOCOL)

        print_green(f"[{episode+1}/{FLAGS.eval_n_trajs}] saved. success={succeed} steps={step_in_ep}")
        episode += 1

    try:
        keyr.stop()
    except Exception:
        pass

    print_green(f"saved eval rollouts to: {record_dir}")
    print_green(f"success rate: {success_counter / float(FLAGS.eval_n_trajs)}")
    print_green(
        f"avg time (success only): {float(np.mean(success_times)):.3f}s"
        if len(success_times)
        else "avg time (success only): nan"
    )


def main(_):
    assert FLAGS.exp_name is not None, "--exp_name is required"
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment name not found in CONFIG_MAPPING"
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    rng = jax.random.PRNGKey(int(FLAGS.seed))
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=False,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

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

    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    print_green("starting evaluation + recording")
    eval_and_record(agent, env, sampling_rng, policy_image_keys=list(config.image_keys))


if __name__ == "__main__":
    app.run(main)
