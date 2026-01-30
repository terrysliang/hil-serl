#!/usr/bin/env python3
"""Evaluate an OpenPI policy checkpoint via WebSocket server using your HIL-SERL env.

This script is intentionally modeled after `eval_record.py`:
- loads an env from `experiments.mappings.CONFIG_MAPPING[exp_name]`
- runs rollouts
- records each kept episode as a pickle
- supports interactive keys: o/c/p/r/q, and keep/drop at end (k/x)

Differences:
- instead of a JAX SAC agent, it queries an OpenPI `serve_policy.py` server via WebSocket.
- it plans action chunks and executes the first `replan_steps` actions before replanning.

Notes
- OpenPI `hilserl_policy.py` expects flat keys:
    side_1, wrist_1_full, wrist_2_full, tcp_pose, tcp_vel, tcp_force, tcp_torque, task.
- This script builds that request dict from the env observation.

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
import numpy as np
import cv2
from collections import deque
from absl import app, flags
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

# --- env / rollout controls ---
flags.DEFINE_string("exp_name", None, "Experiment name (key in CONFIG_MAPPING).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of evaluation episodes.")
flags.DEFINE_integer(
    "eval_max_steps_per_ep",
    0,
    "Optional cap on steps per episode (0 = until termination).",
)
flags.DEFINE_boolean("save_video", False, "Enable env-side video saving if supported.")

flags.DEFINE_string(
    "eval_record_dir",
    None,
    "Directory to save rollouts. Default: <cwd>/openpi_eval_rollouts/<timestamp>",
)
flags.DEFINE_boolean(
    "eval_record_infos",
    False,
    "Whether to record per-step `info` dict (can be large / non-serializable).",
)

flags.DEFINE_string(
    "eval_task",
    "Perform high-precision insertion: align and insert the peg into the hole.",
    "Task description/prompt to send to the policy (stored as `task` by default).",
)

# --- OpenPI server ---
flags.DEFINE_string("host", "127.0.0.1", "OpenPI policy server host.")
flags.DEFINE_integer("port", 8000, "OpenPI policy server port.")
flags.DEFINE_string("api_key", None, "Optional API key for the policy server.")
flags.DEFINE_integer(
    "replan_steps",
    10,
    "How many steps to execute from each predicted action chunk before replanning.",
)

# --- observation mapping ---
flags.DEFINE_list(
    "camera_keys",
    ["side_1", "wrist_1_full", "wrist_2_full"],
    "Camera keys to send to OpenPI (base, left_wrist, right_wrist).",
)
flags.DEFINE_list(
    "proprio_keys",
    ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"],
    "Proprio keys to send to OpenPI.",
)
flags.DEFINE_integer("image_size", 256, "Resize images to this square size before sending.")
flags.DEFINE_boolean(
    "assume_bgr",
    False,
    "If True, convert BGR->RGB before sending (useful if your camera gives OpenCV BGR).",
)
flags.DEFINE_string(
    "task_key",
    "task",
    "Key name used for the instruction string in the request (match your hilserl_policy.py).")

# Recording filter (keep only selected image arrays in stored obs)
flags.DEFINE_list(
    "record_camera_keys",
    ["wrist_1_full", "wrist_2_full", "side_1"],
    "Camera keys to keep under obs['images'] or as top-level in recorded pkl.",
)


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
    """Non-blocking single-key reader for terminal (Linux)."""

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


def _best_effort_release(env):
    """Best-effort cleanup to release camera/robot resources.

    Gymnasium wrappers don't always propagate close() to the underlying env,
    and different camera backends use different teardown methods.
    This function tries several common patterns safely.
    """
    # 1) Close wrapper/env
    try:
        if env is not None and hasattr(env, "close"):
            env.close()
    except Exception:
        pass

    base = None
    try:
        base = _get_base_env(env)
    except Exception:
        base = None

    # 2) Close base env
    try:
        if base is not None and hasattr(base, "close"):
            base.close()
    except Exception:
        pass

    # 3) Release OpenCV windows if any
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # 4) Release common camera handles
    # - Many HIL-SERL envs keep a dict like self.cap[key] where each value has close() or release().
    try:
        cap_dict = getattr(base, "cap", None)
        if isinstance(cap_dict, dict):
            for _, cap in cap_dict.items():
                try:
                    if hasattr(cap, "close"):
                        cap.close()
                    if hasattr(cap, "release"):
                        cap.release()
                    # Some backends wrap a pipeline with stop()
                    if hasattr(cap, "stop"):
                        cap.stop()
                except Exception:
                    pass
    except Exception:
        pass

    # - Some envs store cameras under other names
    for attr in ("camera", "cameras", "side_camera", "wrist_camera", "realsense"):
        try:
            obj = getattr(base, attr, None)
            if obj is None:
                continue
            if isinstance(obj, dict):
                it = obj.values()
            elif isinstance(obj, (list, tuple)):
                it = obj
            else:
                it = [obj]
            for cam in it:
                try:
                    if hasattr(cam, "close"):
                        cam.close()
                    if hasattr(cam, "release"):
                        cam.release()
                    if hasattr(cam, "stop"):
                        cam.stop()
                except Exception:
                    pass
        except Exception:
            pass


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
        """=== OPENPI EVAL CONTROLS ===
   o/c : open/close gripper immediately
   p   : pause/resume env stepping
   r   : drop current episode buffer and reset env immediately
   q   : quit evaluation loop
   (after episode ends) k = keep+save episode, x = drop+redo episode
"""
    )


# ---------------------------
# Obs helpers
# ---------------------------

def _to_numpy_pytree(x):
    """Convert nested structures (dict/list/tuple) of arrays/scalars to numpy arrays."""
    if isinstance(x, dict):
        return {k: _to_numpy_pytree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy_pytree(v) for v in x)
    return np.asarray(x)


def _filter_obs_images(obs_np, camera_keys):
    """Keep only selected image keys when recording."""
    if not camera_keys or not isinstance(obs_np, dict):
        return obs_np

    # Case A: cameras under obs["images"]
    if "images" in obs_np and isinstance(obs_np["images"], dict):
        imgs = obs_np["images"]
        out = dict(obs_np)
        out["images"] = {k: imgs[k] for k in camera_keys if k in imgs}
        return out

    # Case B: top-level image keys
    out = dict(obs_np)
    for k, v in list(out.items()):
        if k in camera_keys:
            continue
        arr = np.asarray(v)
        if isinstance(arr, np.ndarray) and arr.ndim >= 3:
            out.pop(k, None)
    return out


def _get_image_from_obs(obs: dict, key: str):
    """Try to extract image from obs in either nested or top-level form."""
    if not isinstance(obs, dict):
        raise KeyError(f"obs is not a dict; cannot read image '{key}'")

    if "images" in obs and isinstance(obs["images"], dict) and key in obs["images"]:
        return obs["images"][key]

    if key in obs:
        return obs[key]

    # Some wrappers keep only non-_full keys; offer a helpful message.
    raise KeyError(f"Could not find image key '{key}'. Top-level keys={list(obs.keys())}; "
                   f"images keys={list(obs.get('images', {}).keys()) if isinstance(obs.get('images', None), dict) else None}")


def _ensure_uint8_hwc(img: np.ndarray, size: int, assume_bgr: bool) -> np.ndarray:
    """Convert (1,H,W,C) or (C,H,W) or float->uint8 to uint8 HWC; resize to (size,size)."""
    img = np.asarray(img)

    # Drop leading batch dim
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    # float -> uint8
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0, 255).astype(np.uint8)

    # BGR -> RGB if requested
    if assume_bgr and img.ndim == 3 and img.shape[-1] == 3:
        img = img[..., ::-1]

    # Resize
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Image is not HWC(3): shape={img.shape} dtype={img.dtype}")

    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


def _extract_proprio(obs: dict) -> dict:
    """Return dict with tcp_pose/tcp_vel/tcp_force/tcp_torque.

    Supports:
    - obs['state'] as dict with those keys
    - obs['state'] as vector of length 19 (tcp_pose[0:7], tcp_vel[7:13], tcp_force[13:16], tcp_torque[16:19])
    """
    if not isinstance(obs, dict) or "state" not in obs:
        raise KeyError(f"Missing 'state' in obs. keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    st = obs["state"]

    # state dict
    if isinstance(st, dict):
        out = {}
        for k in FLAGS.proprio_keys:
            if k not in st:
                raise KeyError(f"Missing state['{k}'] in obs['state']. Available={list(st.keys())}")
            out[k] = np.asarray(st[k], dtype=np.float32).reshape(-1)
        return out

    # vector form
    vec = np.asarray(st, dtype=np.float32)
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec[0]
    vec = vec.reshape(-1)

    if vec.shape[0] != 19:
        raise ValueError(f"Unsupported state vector length {vec.shape[0]} (expected 19).")

    return {
        "tcp_pose": vec[0:7],
        "tcp_vel": vec[7:13],
        "tcp_force": vec[13:16],
        "tcp_torque": vec[16:19],
    }


def build_openpi_request(obs: dict) -> dict:
    """Build request dict for your OpenPI hilserl_policy.py (flat keys)."""
    base_key, left_key, right_key = list(FLAGS.camera_keys)

    base = _ensure_uint8_hwc(_get_image_from_obs(obs, base_key), FLAGS.image_size, FLAGS.assume_bgr)
    left = _ensure_uint8_hwc(_get_image_from_obs(obs, left_key), FLAGS.image_size, FLAGS.assume_bgr)
    right = _ensure_uint8_hwc(_get_image_from_obs(obs, right_key), FLAGS.image_size, FLAGS.assume_bgr)

    proprio = _extract_proprio(obs)

    req = {
        base_key: base,
        left_key: left,
        right_key: right,
        FLAGS.task_key: FLAGS.eval_task,
    }
    req.update(proprio)
    return req


def _env_rate_info(env):
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


def eval_and_record_openpi(env):
    # Import here so your conda env can decide how to install it.
    try:
        from openpi_client import websocket_client_policy as _websocket_client_policy
    except Exception as e:
        raise RuntimeError(
            "openpi_client is not available in this environment. "
            "Install it (in your conda env) with: pip install -U openpi-client\n"
            f"Original import error: {e}"
        )

    client = _websocket_client_policy.WebsocketClientPolicy(
        host=FLAGS.host,
        port=FLAGS.port,
        api_key=FLAGS.api_key,
    )

    print_green(f"[openpi_client] connected. server_metadata={client.get_server_metadata()}")

    # Warm up to force model load
    print_green("[openpi_client] warm-up...")
    dummy = {
        FLAGS.camera_keys[0]: np.zeros((FLAGS.image_size, FLAGS.image_size, 3), dtype=np.uint8),
        FLAGS.camera_keys[1]: np.zeros((FLAGS.image_size, FLAGS.image_size, 3), dtype=np.uint8),
        FLAGS.camera_keys[2]: np.zeros((FLAGS.image_size, FLAGS.image_size, 3), dtype=np.uint8),
        "tcp_pose": np.zeros((7,), dtype=np.float32),
        "tcp_vel": np.zeros((6,), dtype=np.float32),
        "tcp_force": np.zeros((3,), dtype=np.float32),
        "tcp_torque": np.zeros((3,), dtype=np.float32),
        FLAGS.task_key: FLAGS.eval_task,
    }
    _ = client.infer(dummy)
    print_green("[openpi_client] warm-up done")

    # record dir
    if FLAGS.eval_record_dir is not None:
        record_dir = FLAGS.eval_record_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = os.path.join(os.getcwd(), "openpi_eval_rollouts", ts)
    os.makedirs(record_dir, exist_ok=True)

    hz, dt, action_scale = _env_rate_info(env)
    meta = dict(
        exp_name=FLAGS.exp_name,
        seed=int(FLAGS.seed),
        eval_n_trajs=int(FLAGS.eval_n_trajs),
        eval_task=str(FLAGS.eval_task),
        control_hz=hz,
        dt=dt,
        action_semantics="delta_pose",
        action_scale=action_scale,
        camera_keys=list(FLAGS.camera_keys),
        proprio_keys=list(FLAGS.proprio_keys),
        image_size=int(FLAGS.image_size),
        server_host=str(FLAGS.host),
        server_port=int(FLAGS.port),
        recorded_at=datetime.datetime.now().isoformat(),
    )

    # interactive controls
    keyr = KeyReader()
    keyr.start()
    _print_controls()

    success_counter = 0.0
    success_times = []

    episode = 0
    pending_reset = None
    while episode < int(FLAGS.eval_n_trajs):
        if pending_reset is None:
            obs, _ = env.reset()
        else:
            obs, _ = pending_reset
            pending_reset = None

        done = False
        truncated = False
        start_time = time.time()
        step_in_ep = 0
        last_info = {}

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

        base_env = _get_base_env(env)
        paused = False
        redo_episode = False
        quit_all = False

        action_plan: deque[np.ndarray] = deque()

        while not (done or truncated):
            # Handle keys
            ch = keyr.get_key(timeout_s=0.0)
            if ch:
                ch = ch.lower()
                if ch == "p":
                    paused = not paused
                    print_green(f"[openpi_eval] paused={paused}")
                elif ch == "o":
                    if _robot_open(base_env):
                        print_green("[openpi_eval] gripper_open sent")
                elif ch == "c":
                    if _robot_close(base_env):
                        print_green("[openpi_eval] gripper_close sent")
                elif ch == "r":
                    print_green("[openpi_eval] dropping current episode + resetting env")
                    redo_episode = True
                    break
                elif ch == "q":
                    print_green("[openpi_eval] quitting")
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

            # Replan if needed
            if not action_plan:
                req = build_openpi_request(obs)
                out = client.infer(req)
                if "actions" not in out:
                    raise KeyError(f"OpenPI server response missing 'actions'. keys={list(out.keys())}")
                action_chunk = np.asarray(out["actions"], dtype=np.float32)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]
                if action_chunk.shape[0] < int(FLAGS.replan_steps):
                    raise ValueError(
                        f"Need replan_steps={FLAGS.replan_steps}, but policy returned only {action_chunk.shape[0]} steps"
                    )
                for a in action_chunk[: int(FLAGS.replan_steps)]:
                    action_plan.append(np.asarray(a, dtype=np.float32))

            action = action_plan.popleft()

            # Match env action dim if needed (e.g., env expects 7 but policy outputs 6)
            act_dim = int(np.prod(getattr(env.action_space, "shape", (len(action),))))
            if action.shape[-1] != act_dim:
                if action.shape[-1] == 6 and act_dim == 7:
                    action = np.concatenate([action, np.zeros((1,), dtype=np.float32)], axis=0)
                else:
                    raise ValueError(f"Action dim mismatch: policy={action.shape[-1]} env={act_dim}")

            next_obs, reward, done, truncated_step, info = env.step(action)
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
            ep["actions"].append(np.asarray(action))
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
                        print_green("[openpi_eval] gripper_open sent")
                elif ch == "c":
                    if _robot_close(base_env):
                        print_green("[openpi_eval] gripper_close sent")
                elif ch == "r":
                    decision = "drop"

        if decision == "quit":
            break

        if decision == "drop":
            print_green("[openpi_eval] dropped episode; redoing same index")
            try:
                pending_reset = env.reset()
            except Exception:
                pending_reset = None
            continue

        # keep
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
        f"avg time (success only): {float(np.mean(success_times)):.3f}s" if len(success_times) else "avg time (success only): nan"
    )


def main(_):
    assert FLAGS.exp_name is not None, "--exp_name is required"
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment name '{FLAGS.exp_name}' not found in CONFIG_MAPPING"

    config = CONFIG_MAPPING[FLAGS.exp_name]()

    env = config.get_environment(
        fake_env=False,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    print_green("starting OpenPI evaluation + recording")
    try:
        eval_and_record_openpi(env)
    finally:
        # Ensure cameras (esp. side camera) are released even on quit/exception.
        _best_effort_release(env)


if __name__ == "__main__":
    app.run(main)
