#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from pynput import keyboard

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "/home/terry/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
print(f"JAX cache dir: {jax.config.values['jax_compilation_cache_dir']}")
print(f"JAX cache enabled: {jax.config.values.get('jax_enable_compilation_cache', 'Not set')}")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

# ---------------- Actor keyboard controls ----------------
# These only affect the ACTOR loop.
#   o/c : open/close gripper immediately
#   p   : pause/resume env stepping (use while manually adjusting peg)
#   r   : drop current episode buffer and reset env immediately
#   q   : quit actor loop gracefully
#   (after episode ends) k = keep episode, x = drop episode
KEY_OPEN_GRIPPER = 'o'
KEY_CLOSE_GRIPPER = 'c'
KEY_TOGGLE_PAUSE = 'p'
KEY_RESET = 'r'
KEY_QUIT = 'q'
KEY_KEEP_EPISODE = 'k'
KEY_DROP_EPISODE = 'x'

# Shared flags updated by keyboard thread
_actor_pause = False
_actor_quit = False
_actor_reset_req = False
_actor_gripper_req = None  # 'open' or 'close'
_actor_episode_decision = None  # 'keep' or 'drop' when waiting for decision


def _as_char(key):
    try:
        return key.char
    except Exception:
        return None


def _matches_key(key, target_char: str) -> bool:
    ch = _as_char(key)
    return ch == target_char


def _unwrap_env(env):
    """Walk through gymnasium wrappers to reach the base env."""
    visited = set()
    cur = env
    for _ in range(50):
        if cur is None or id(cur) in visited:
            break
        visited.add(id(cur))
        yield cur
        # gymnasium wrappers expose .env
        if hasattr(cur, 'env'):
            nxt = getattr(cur, 'env')
            if nxt is not None and nxt is not cur:
                cur = nxt
                continue
        # gymnasium has .unwrapped
        if hasattr(cur, 'unwrapped'):
            try:
                nxt = cur.unwrapped
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    continue
            except Exception:
                pass
        break


def _get_base_env(env):
    last = env
    for e in _unwrap_env(env):
        last = e
    return last


def _send_gripper(env, open_or_close: str) -> bool:
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


def _actor_on_press(key):
    """Keyboard handler for actor loop."""
    global _actor_pause, _actor_quit, _actor_reset_req
    global _actor_gripper_req, _actor_episode_decision

    if _matches_key(key, KEY_QUIT):
        _actor_quit = True
        return

    if _matches_key(key, KEY_TOGGLE_PAUSE):
        _actor_pause = not _actor_pause
        return

    if _matches_key(key, KEY_RESET):
        _actor_reset_req = True
        return

    if _matches_key(key, KEY_OPEN_GRIPPER):
        _actor_gripper_req = 'open'
        return

    if _matches_key(key, KEY_CLOSE_GRIPPER):
        _actor_gripper_req = 'close'
        return

    if _matches_key(key, KEY_KEEP_EPISODE):
        _actor_episode_decision = 'keep'
        return

    if _matches_key(key, KEY_DROP_EPISODE):
        _actor_episode_decision = 'drop'
        return

# ---------------------------------------------------------


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    # actor keyboard listener
    print_green("\n=== ACTOR CONTROLS ===")
    print_green("  o/c : open/close gripper immediately")
    print_green("  p   : pause/resume env stepping (safe while adjusting peg)")
    print_green("  r   : drop current episode buffer and reset env immediately")
    print_green("  q   : quit actor loop")
    print_green("  (after episode ends) k = keep episode, x = drop episode\n")

    listener = keyboard.Listener(on_press=_actor_on_press)
    listener.start()

    # episode buffers (only committed if you choose KEEP)
    ep_transitions = []
    ep_demo_transitions = []

    # progress bar counts environment steps only
    pbar = tqdm.tqdm(total=config.max_steps, initial=start_step, dynamic_ncols=True)
    step = start_step

    try:
        while step < config.max_steps:

            # Always pull latest params (esp. during pauses)
            client.update()

            # Quit
            if _actor_quit:
                print_green("[actor] quit requested; exiting.")
                break

            # Apply gripper command if requested (works even when paused)
            if _actor_gripper_req is not None:
                req = _actor_gripper_req
                # clear the request early to avoid repeats
                globals()['_actor_gripper_req'] = None
                ok = _send_gripper(env, req)
                if not ok:
                    print("[warn] gripper command failed (no method/endpoint reachable).")

            # Reset request: drop current episode and reset immediately
            if _actor_reset_req:
                globals()['_actor_reset_req'] = False
                ep_transitions = []
                ep_demo_transitions = []
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                obs, _ = env.reset()
                continue

            # Pause: do not step env (do not send pose commands)
            if _actor_pause:
                time.sleep(0.05)
                # update the progress bar description while paused
                pbar.set_description(f"paused | last return={running_return:.2f}")
                continue
            
            timer.tick("total")
            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=True,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                running_return += reward

                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty'] = info['grasp_penalty']

                ep_transitions.append(copy.deepcopy(transition))
                if already_intervened:
                    ep_demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                step += 1
                pbar.update(1)

                # Episode end: ask keep/drop
                if done or truncated:
                    # add intervention stats to RecordEpisodeStatistics payload if present
                    if isinstance(info, dict) and "episode" in info and isinstance(info["episode"], dict):
                        info["episode"]["intervention_count"] = intervention_count
                        info["episode"]["intervention_steps"] = intervention_steps

                    # Show summary and wait for decision
                    globals()['_actor_episode_decision'] = None
                    print_green("\n[episode ended]")
                    print_green(f"  return={running_return:.3f}  steps={len(ep_transitions)}  intervened_eps={intervention_count}  intervene_steps={intervention_steps}")
                    print_green("  press 'k' to KEEP episode, 'x' to DROP episode (you can also press 'r' to reset).\n")

                    # Wait for decision (you can still open/close gripper while waiting)
                    while globals().get('_actor_episode_decision', None) is None and (not globals().get('_actor_quit', False)):
                        # handle gripper during decision wait
                        if globals().get('_actor_gripper_req', None) is not None:
                            req = globals()['_actor_gripper_req']
                            globals()['_actor_gripper_req'] = None
                            _send_gripper(env, req)
                        if globals().get('_actor_reset_req', False):
                            globals()['_actor_reset_req'] = False
                            globals()['_actor_episode_decision'] = 'drop'
                            break
                        time.sleep(0.05)

                    if _actor_quit:
                        break

                    decision = globals().get('_actor_episode_decision', 'keep')
                    kept = (decision == 'keep')

                    if kept and len(ep_transitions) > 0:
                        # commit episode transitions to learner
                        for t in ep_transitions:
                            data_store.insert(t)
                            transitions.append(copy.deepcopy(t))
                        for t in ep_demo_transitions:
                            intvn_data_store.insert(t)
                            demo_transitions.append(copy.deepcopy(t))

                        # send stats to learner for logging ONLY if kept
                        stats = {"environment": info}
                        client.request("send-stats", stats)
                        pbar.set_description(f"kept | last return: {running_return:.2f}")
                    else:
                        pbar.set_description(f"dropped | last return: {running_return:.2f}")

                    # reset for next episode
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    ep_transitions = []
                    ep_demo_transitions = []
                    obs, _ = env.reset()
                # client.request("send-stats", stats)

            # periodic buffer dump (only includes KEPT episodes)
            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                os.makedirs(buffer_path, exist_ok=True)
                os.makedirs(demo_buffer_path, exist_ok=True)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []
            
            timer.tock("total")
    finally:
        # final dump on exit (only if there are pending kept episodes not yet dumped)
        if FLAGS.checkpoint_path is not None and step > 0 and config.buffer_period > 0 and (step % config.buffer_period) != 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            if len(transitions) > 0:
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
            if len(demo_transitions) > 0:
                with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(demo_transitions, f)

        try:
            listener.stop()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
