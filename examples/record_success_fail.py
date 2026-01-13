import copy
import os
import time
import datetime
import pickle as pkl
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")
flags.DEFINE_integer("failures_needed", 0, "If >0, also stop after collecting this many failures.")

# Episode limit handling
flags.DEFINE_bool(
    "disable_episode_limit",
    True,
    "Disable FrankaEnv/MAX_EPISODE_LENGTH autoreset by setting env.max_episode_length huge and ignoring done/truncated unless user requests reset.",
)
flags.DEFINE_integer("episode_limit_huge", 10**9, "Value to use when disabling episode length limit.")

# Observation mode
flags.DEFINE_bool(
    "passive_observe",
    False,
    "If True, do NOT call env.step(); only poll env._update_currpos() + env._get_obs(). "
    "This avoids sending /pose commands, useful if you move the robot with another tool/script.",
)
flags.DEFINE_float("poll_hz", 10.0, "Hz for passive observation (and for pause-mode observation).")

# --- Keyboard mapping ---
KEY_TOGGLE_SUCCESS = keyboard.Key.space  # toggle success recording
KEY_TOGGLE_FAILURE = "f"                # toggle failure recording
KEY_OPEN_GRIPPER = "o"
KEY_CLOSE_GRIPPER = "c"
KEY_RESET_ROBOT = "r"                   # env.reset() (go to reset pose)
KEY_END_EPISODE = keyboard.Key.esc      # end current episode (env.reset())
KEY_PAUSE = "p"                         # pause/resume (pause => observe-only, no motion commands)
KEY_QUIT = "q"                          # quit & save


label_mode = "ignore"  # one of: ignore / success / failure
pause_mode = False
done_key = False
quit_key = False

_gripper_request: Optional[str] = None  # "open" / "close"
_reset_requested = False


def _as_char(key):
    try:
        return key.char
    except Exception:
        return None


def _matches_key(key, target):
    if isinstance(target, keyboard.Key):
        return key == target
    ch = _as_char(key)
    return ch == target


def on_press(key):
    global label_mode, pause_mode, done_key, quit_key
    global _gripper_request, _reset_requested

    if _matches_key(key, KEY_QUIT):
        quit_key = True
        return

    if _matches_key(key, KEY_PAUSE):
        pause_mode = not pause_mode
        return

    if _matches_key(key, KEY_END_EPISODE):
        done_key = True
        return

    if _matches_key(key, KEY_RESET_ROBOT):
        _reset_requested = True
        return

    if _matches_key(key, KEY_TOGGLE_SUCCESS):
        label_mode = "ignore" if label_mode == "success" else "success"
        return

    if _matches_key(key, KEY_TOGGLE_FAILURE):
        label_mode = "ignore" if label_mode == "failure" else "failure"
        return

    if _matches_key(key, KEY_OPEN_GRIPPER):
        _gripper_request = "open"
        return

    if _matches_key(key, KEY_CLOSE_GRIPPER):
        _gripper_request = "close"
        return


def _save_dataset(exp_name: str, successes, failures):
    os.makedirs("./classifier_data", exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if len(successes) > 0:
        file_name = f"./classifier_data/{exp_name}_{len(successes)}_success_transitions_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(successes, f)
        print(f"[saved] {len(successes)} success transitions -> {file_name}")

    if len(failures) > 0:
        file_name = f"./classifier_data/{exp_name}_{len(failures)}_failure_transitions_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(failures, f)
        print(f"[saved] {len(failures)} failure transitions -> {file_name}")


def _disable_episode_limit(env) -> None:
    """FrankaEnv uses an internal counter: done = curr_path_length >= max_episode_length ..."""
    huge = int(FLAGS.episode_limit_huge)
    for name in ["max_episode_length", "_max_episode_length", "max_path_length", "_max_path_length", "horizon", "_horizon"]:
        if hasattr(env, name):
            try:
                setattr(env, name, huge)
                print(f"[info] set {env.__class__.__name__}.{name} = {huge}")
            except Exception:
                pass


def _gripper_open_close(env, open_or_close: str) -> bool:
    """
    Prefer FrankaEnv internal helper to keep consistent with its safety gating and gripper_sleep.
    - open  => env._send_gripper_command(+1.0)
    - close => env._send_gripper_command(-1.0)
    Falls back to direct POST if needed.
    """
    # Ensure we have fresh curr_gripper_pos for gating
    if hasattr(env, "_update_currpos"):
        try:
            env._update_currpos()
        except Exception:
            pass

    if hasattr(env, "_send_gripper_command"):
        try:
            env._send_gripper_command(1.0 if open_or_close == "open" else -1.0)
            return True
        except Exception:
            pass

    # Fallback: direct endpoint (no gating)
    url = getattr(env, "url", None)
    if isinstance(url, str) and len(url) > 0:
        import requests
        try:
            ep = "open_gripper" if open_or_close == "open" else "close_gripper"
            requests.post(url + ep)
            return True
        except Exception:
            pass

    return False


def _observe_only(env) -> Dict[str, Any]:
    """
    Poll state+images without sending any motion/gripper commands.
    Returns a dict matching the (next_obs, rew, done, truncated, info) pattern.
    """
    if hasattr(env, "_update_currpos"):
        env._update_currpos()
    next_obs = env._get_obs() if hasattr(env, "_get_obs") else env.get_obs()  # best-effort
    return dict(next_obs=next_obs, rew=0, done=False, truncated=False, info={})


def main(_):
    global label_mode, pause_mode, done_key, quit_key
    global _gripper_request, _reset_requested

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    if FLAGS.disable_episode_limit and not FLAGS.passive_observe:
        _disable_episode_limit(env)

    print("\n=== Controls ===")
    print("  space : toggle SUCCESS recording (success / ignore)")
    print("  f     : toggle FAILURE recording (failure / ignore)")
    print("  o/c   : open/close gripper (via FrankaEnv -> /open_gripper,/close_gripper)")
    print("  r     : reset to reset pose (env.reset())")
    print("  esc   : end episode (env.reset())")
    print("  p     : pause/resume (pause => observe-only, no motion commands)")
    print("  q     : quit & save\n")
    if FLAGS.passive_observe:
        print("[mode] passive_observe=True: will NOT call env.step(); only polls state+images.\n")

    obs, _ = env.reset()

    successes = []
    failures = []
    success_needed = int(FLAGS.successes_needed)
    failure_needed = int(FLAGS.failures_needed)

    pbar = tqdm(total=success_needed, desc="successes")
    last_status_time = time.time()

    try:
        while True:
            if quit_key:
                break

            # Reset request (handled in main thread)
            if _reset_requested:
                _reset_requested = False
                label_mode = "ignore"
                pause_mode = False
                done_key = False
                _gripper_request = None
                obs, _ = env.reset()

            # Gripper request should still work even in pause mode
            if _gripper_request is not None:
                req = _gripper_request
                _gripper_request = None
                ok = _gripper_open_close(env, req)
                if not ok:
                    print("[warn] gripper command failed (no method/endpoint reachable).")

            # Episode end request
            if done_key:
                done_key = False
                label_mode = "ignore"
                pause_mode = False
                obs, _ = env.reset()

            # Decide whether to step or just observe
            if FLAGS.passive_observe or pause_mode:
                out = _observe_only(env)
                next_obs, rew, done, truncated, info = out["next_obs"], out["rew"], out["done"], out["truncated"], out["info"]
                # throttle
                hz = float(FLAGS.poll_hz) if FLAGS.poll_hz > 0 else 10.0
                time.sleep(1.0 / hz)
                executed_action = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            else:
                actions = np.zeros(env.action_space.sample().shape, dtype=np.float32)
                next_obs, rew, done, truncated, info = env.step(actions)
                executed_action = actions
                if isinstance(info, dict) and "intervene_action" in info:
                    executed_action = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=executed_action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - float(done),
                    dones=done,
                )
            )
            obs = next_obs

            # Only record when explicitly toggled
            if label_mode == "success":
                successes.append(transition)
                pbar.update(1)
            elif label_mode == "failure":
                failures.append(transition)

            # Periodic status
            now = time.time()
            if now - last_status_time > 2.0:
                last_status_time = now
                tqdm.write(
                    f"[status] mode={label_mode:7s}  pause={pause_mode}  successes={len(successes)}/{success_needed}  failures={len(failures)}"
                )

            # Stop condition
            if len(successes) >= success_needed and (failure_needed <= 0 or len(failures) >= failure_needed):
                break

            # Autoreset removal: ignore done/truncated unless user asked reset
            # (If you *do* want autoreset, just run with --disable_episode_limit=False and use esc/r as before.)
            if (done or truncated) and (not FLAGS.disable_episode_limit):
                label_mode = "ignore"
                pause_mode = False
                obs, _ = env.reset()

    finally:
        try:
            pbar.close()
        except Exception:
            pass

        _save_dataset(FLAGS.exp_name, successes, failures)

        try:
            listener.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

        
if __name__ == "__main__":
    app.run(main)
