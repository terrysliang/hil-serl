import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.sheath_insertion.wrapper import SheathEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "230322274751",
            "dim": (1280, 720),
            "exposure": 5000,
            "contrast": 77,
        },
        "wrist_2": {
            "serial_number": "218622278628",
            "dim": (1280, 720),
            "exposure": 5000,
            "contrast": 77,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:550, 490:890],
        "wrist_2": lambda img: img[150:550, 480:880],
    }
    TARGET_POSE = np.array([0.13387135, 0.55630949, 0.11354294, np.pi, 0, 0])
    GRASP_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0, 0])
    # RESET_POSE =  np.array([0.13495839, 0.55607059, 0.16173811, np.pi, 0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.05, 0.05, 0.01, 0.3, 0.3, 0.3])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.05, 0.08, 0.3, 0.3, 0.3])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.02, 0.06, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 200
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 3000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = SheathEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                # classifier(obs) can be shape (1,) or (T,) depending on wrappers/chunking
                p = jax.nn.sigmoid(classifier(obs))
                p_scalar = float(jnp.ravel(p)[-1])            # <-- scalar python float

                jaw = float(obs["state"][-1, 6])
                state_last = jnp.asarray(obs["state"])[-1]
                print(f"p={p_scalar:.4f}, jaw={jaw:.4f}, reward={(p_scalar>0.85) and (jaw>0.025)}")
                return int((p_scalar > 0.85) and (jaw > 0.049))

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env