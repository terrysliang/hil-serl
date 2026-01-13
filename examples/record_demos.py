import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

# --- Controls ---
# i: toggle ignore/record (default: IGNORE so you can set up the initial state)
KEY_TOGGLE_RECORD = "i"
KEY_OPEN_GRIPPER = "o"
KEY_CLOSE_GRIPPER = "c"
KEY_RESET = "r"              # reset env (go to reset pose)
KEY_END_EPISODE = keyboard.Key.esc  # end episode (reset env)
KEY_QUIT = "q"               # quit and save what you have so far

record_enabled = False
done_key = False
quit_key = False
_gripper_request = None
_reset_requested = False


def _as_char(key):
    try:
        return key.char
    except Exception:
        return None


def _matches_key(key, target):
    if isinstance(target, keyboard.Key):
        return key == target
    return _as_char(key) == target


def on_press(key):
    global record_enabled, done_key, quit_key, _gripper_request, _reset_requested

    if _matches_key(key, KEY_QUIT):
        quit_key = True
        return

    if _matches_key(key, KEY_END_EPISODE):
        done_key = True
        return

    if _matches_key(key, KEY_RESET):
        _reset_requested = True
        return

    if _matches_key(key, KEY_TOGGLE_RECORD):
        record_enabled = not record_enabled
        return

    if _matches_key(key, KEY_OPEN_GRIPPER):
        _gripper_request = "open"
        return

    if _matches_key(key, KEY_CLOSE_GRIPPER):
        _gripper_request = "close"
        return


def _send_gripper(env, open_or_close: str) -> bool:
    """
    Prefer FrankaEnv internal helper to keep consistent with its safety gating and gripper_sleep.
    - open  => env._send_gripper_command(+1.0)
    - close => env._send_gripper_command(-1.0)
    Fallback: direct POST if env.url exists.
    """
    # keep curr_gripper_pos fresh for gating
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

    url = getattr(env, "url", None)
    if isinstance(url, str) and url:
        import requests
        try:
            ep = "open_gripper" if open_or_close == "open" else "close_gripper"
            requests.post(url + ep)
            return True
        except Exception:
            pass

    return False


def main(_):
    global record_enabled, done_key, quit_key, _gripper_request, _reset_requested

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    print("\n=== Controls ===")
    print("  i     : toggle RECORD / IGNORE (default IGNORE)")
    print("  o/c   : open/close gripper")
    print("  r     : reset to reset pose (env.reset())")
    print("  esc   : end episode (env.reset())")
    print("  q     : quit & save\n")

    obs, info = env.reset()
    print("Reset done. Mode=IGNORE (press 'i' to start recording a demo from *current* state).")

    transitions = []
    success_count = 0
    success_needed = int(FLAGS.successes_needed)
    pbar = tqdm(total=success_needed, desc="success_demos")

    # per-episode buffer
    trajectory = []
    returns = 0.0

    record_latch = record_enabled  # detect IGNORE->RECORD transitions

    try:
        while success_count < success_needed and not quit_key:
            # handle reset requests
            if _reset_requested:
                _reset_requested = False
                record_enabled = False
                trajectory = []
                returns = 0.0
                obs, info = env.reset()
                continue

            # gripper requests (work even in IGNORE mode)
            if _gripper_request is not None:
                req = _gripper_request
                _gripper_request = None
                ok = _send_gripper(env, req)
                if not ok:
                    print("[warn] gripper command failed (no method/endpoint reachable).")

            # end-episode key
            if done_key:
                done_key = False
                record_enabled = False
                trajectory = []
                returns = 0.0
                obs, info = env.reset()
                continue

            # If user just turned recording ON, start a *fresh* trajectory from current state.
            # (This avoids capturing your manual setup motions as part of the demo.)
            # We detect this by checking a marker in info dict.
            # Detect IGNORE->RECORD toggle mid-episode: start a fresh trajectory from current state.
            if record_enabled and (not record_latch):
                trajectory = []
                returns = 0.0
            record_latch = record_enabled

            # Step env (teleop is handled by env itself; we pass zeros)
            actions = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += float(rew)

            # Use the real executed action if provided
            if isinstance(info, dict) and "intervene_action" in info:
                actions = info["intervene_action"]

            # Only store transitions when RECORD is enabled
            if record_enabled:
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - float(done),
                        dones=done,
                        infos=info,
                    )
                )
                trajectory.append(transition)

            # UI
            mode = "RECORD" if record_enabled else "IGNORE"
            pbar.set_description(f"mode={mode} return={returns:.1f}")

            obs = next_obs

            # If episode ended (by env success/limit/terminate), decide whether to keep it.
            if done or truncated:
                succeed = False
                if isinstance(info, dict):
                    succeed = bool(info.get("succeed", False))

                if succeed and len(trajectory) > 0:
                    transitions.extend(copy.deepcopy(trajectory))
                    success_count += 1
                    pbar.update(1)
                    print(f"[demo] saved one success demo (#{success_count}/{success_needed}), steps={len(trajectory)}, return={returns:.1f}")

                # reset for next attempt
                record_enabled = False
                trajectory = []
                returns = 0.0
                obs, info = env.reset()

            # clear current trajectory and restart counting returns from that moment.
            # We do this by checking the record_enabled flag and a local latch.
            # (Latch lives as a local variable.)
            # NOTE: implemented below with a simple check.
            # This logic needs a latch:
            # We'll keep it outside the loop:

    finally:
        try:
            pbar.close()
        except Exception:
            pass

        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")

        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_count}_demos_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
        print(f"[saved] {success_count} demos -> {file_name}")

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