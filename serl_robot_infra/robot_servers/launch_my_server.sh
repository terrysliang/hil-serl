# Source the setup.bash file for the first ROS workspace
source /workspace/hil-serl/src/catkin_ws/devel/setup.bash

# Set ROS master URI to localhost
export ROS_MASTER_URI=http://localhost:11311

# Run the first instance of franka_server.py in the background
/usr/bin/python3 /workspace/hil-serl/src/hil-serl/serl_robot_infra/robot_servers/franka_server.py \
    --robot_ip=172.16.0.3 \
    --gripper_type=DH \
    --reset_joint_target=1.5959619886,0.1452558658,-0.2614405266,-2.0465805071,0.0455929640,2.1850543402,1.3107662306 \
    --flask_url=127.0.0.1 \
    --ros_port=11311
