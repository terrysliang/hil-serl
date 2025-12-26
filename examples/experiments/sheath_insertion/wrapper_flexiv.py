import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
import gymnasium as gym

from env.flexiv.flexiv_env import FlexivEnv

class TaskEnv(FlexivEnv):
    def __init__(self, eval_env = False,
                    **kwargs):
        super().__init__(**kwargs)

        # self.eval_env = eval_env
        self.eval_env = kwargs.get('eval_env', False)
        fake_env = kwargs.get('fake_env', False)

        self.is_reset_grab = True
        self.should_grasp = False

        if eval_env:
            """ 验证模式下，无需重新抓取 """
            self.is_reset_grab = False

        if not fake_env:
            self.start_key_listener()

    def start_key_listener(self):
        from pynput import keyboard

        # 键盘控制具体逻辑：1）控制重新抓取
        def on_press(key):
            if str(key) == "Key.f2":
                self.should_grasp = True
            elif str(key) == "Key.f3":
                self.is_reset_grab = not self.is_reset_grab
                print(">>> is_reset_grab = ", self.is_reset_grab)

        self.listener = keyboard.Listener(
            on_press=on_press)
        self.listener.start()

    def stop_key_listener(self):
        self.listener.stop()

    def _open_gripper(self):
        """ 能松开抓取物体的最小开度 """
        self._send_gripper_command(20, force=100, mode="continuous")

    def _close_gripper(self):
        self._send_gripper_command(0, force=100, mode="continuous")


    def go_to_reset(self, joint_reset=False):
        # 退出力控
        self._stop_imp_control()

        if self.is_reset_grab:
            self._open_gripper()

            # 向上退出
            self._update_currstate()
            pose = copy.deepcopy(self.currpos)  # quat
            pose[2] += 0.02
            self._send_target_pose(pose, speed_rate=0.2)

            # 运动到抓取点
            self._send_target_joints(self.config.TASK_GRAB_JOINTS, speed_rate=0.5)
            
            # 等待抓取命令
            print("press 'f2' to grasp and continue")
            while not self.should_grasp:
                time.sleep(0.04)
            self.should_grasp = False
            self._close_gripper()
            time.sleep(1)


        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.config.TASK_INIT_POSE.copy()
            
            reset_pose[:3] += np.random.uniform(
                -self.config.RANDOM_XYZ_RANGE, self.config.RANDOM_XYZ_RANGE, (3,)
            )
            euler_random = self.config.TASK_INIT_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_random
            # print(f"reset_pose : {reset_pose}")

            self._send_target_pose(reset_pose, speed_rate=0.3)
        else:
            # 运动到固定任务起始点
            self._send_target_pose(self.config.TASK_INIT_POSE, speed_rate=0.3)

        self._start_imp_control()
        self._set_impefance_param(self.config.IMPEDANCE_PARAM)