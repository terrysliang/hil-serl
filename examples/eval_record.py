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

# JAX sharding (keep consistent with training code path)
devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x: str):
    print("\033[92m {}\033[00m".format(x))


def _to_numpy_pytree(x):
    """Convert nested structures (dict/list/tuple) of arrays/scalars to numpy arrays."""
    return jax.tree_map(lambda y: np.asarray(y), x)


def _filter_obs_images(obs_np, camera_keys):
    """
    Keep only selected camera keys under obs['images'] to reduce file size.
    Safe no-op if structure doesn't match.
    """
    if not camera_keys:
        return obs_np
    if isinstance(obs_np, dict) and "images" in obs_np and isinstance(obs_np["images"], dict):
        imgs = obs_np["images"]
        missing = [k for k in camera_keys if k not in imgs]
        if missing:
            print_green(f"[eval_record] warning: missing cameras in obs['images']: {missing}")
        obs_np = dict(obs_np)
        obs_np["images"] = {k: imgs[k] for k in camera_keys if k in imgs}
    return obs_np


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


def eval_and_record(agent, env, rng):
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
        camera_keys=list(FLAGS.eval_camera_keys),
        task=FLAGS.eval_task,
        recorded_at=datetime.datetime.now().isoformat(),
    )

    success_counter = 0.0
    success_times = []

    for episode in range(int(FLAGS.eval_n_trajs)):
        obs, _ = env.reset()
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

        while not (done or truncated):
            if FLAGS.eval_max_steps_per_ep and step_in_ep >= FLAGS.eval_max_steps_per_ep:
                truncated = True
                break

            rng, key = jax.random.split(rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=False, #TODO
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated_step, info = env.step(actions)
            last_info = info
            truncated = truncated or bool(truncated_step)

            obs_np = _filter_obs_images(_to_numpy_pytree(obs), FLAGS.eval_camera_keys)
            next_obs_np = _filter_obs_images(_to_numpy_pytree(next_obs), FLAGS.eval_camera_keys)

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

        ep["final_observation"] = _filter_obs_images(_to_numpy_pytree(obs), FLAGS.eval_camera_keys)
        ep["num_steps"] = int(step_in_ep)

        succeed = False
        if isinstance(last_info, dict) and "succeed" in last_info:
            succeed = bool(last_info.get("succeed"))
        elif len(ep["rewards"]) > 0:
            succeed = bool(ep["rewards"][-1])
        ep["success"] = succeed

        if succeed:
            success_times.append(time.time() - start_time)
        success_counter += float(succeed)

        out_path = os.path.join(record_dir, f"episode_{episode:04d}.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(ep, f, protocol=pkl.HIGHEST_PROTOCOL)

        print_green(f"[{episode+1}/{FLAGS.eval_n_trajs}] success={succeed} steps={step_in_ep}")

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
    eval_and_record(agent, env, sampling_rng)


if __name__ == "__main__":
    app.run(main)
