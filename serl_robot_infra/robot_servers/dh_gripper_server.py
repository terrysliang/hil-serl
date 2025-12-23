import time
from robot_servers.gripper_server import GripperServer
from dh_gripper import DHGripper
import time
import rospy
from std_msgs.msg import String

class DHGripperServer(GripperServer):
    def __init__(self, gripper_port="/dev/ttyUSB0"):
        super().__init__()
        self.gripper = DHGripper(port=gripper_port)
        self.state_sub = rospy.Subscriber("simple_msg", String, self._update_gripper)
        self.gripper_state = 0


    def _update_gripper(self, msg):
        self.gripper_pos = self.gripper.read_pos() / 1000.0
        self.gripper_state = self.gripper.read_state()


    def activate_gripper(self):
        self.gripper.init_state()
        time.sleep(2)

    def reset_gripper(self):
        self.gripper.init_state()
        time.sleep(2)

    def open(self):
        self.gripper.set_vel(1000)
        self.gripper.set_pos(1000)

    def close(self):
        self.gripper.set_force(40)
        self.gripper.set_vel(1000)
        self.gripper.set_pos(0)

    def move(self, position): # 0~1000: 0~100%
        self.gripper.set_force(40)
        self.gripper.set_vel(1000)
        self.gripper.set_pos(position)

    def close_slow(self):
        self.gripper.set_force(40)
        self.gripper.set_vel(50)
        self.gripper.set_pos(0)



if __name__ == "__main__":
    gripper_server = DHGripperServer("/dev/ttyUSB1")
    # gripper_server.activate_gripper()
    # gripper_server.open()
    # time.sleep(2)
    # gripper_server.close()
    # time.sleep(2)
    gripper_server.move(450)
    gripper_server.move(600)
    time.sleep(2)
    # gripper_server.move(500)
    # time.sleep(2)
    # gripper_server.close_slow()
    # time.sleep(2)
    # gripper_server.reset_gripper()
    