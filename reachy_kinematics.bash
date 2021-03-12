# Run "ros2 launch reachy_kinematics kinematics.launch.py"

# source ROS2 Foxy setup file.
# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash
source /home/nuc/reachy_ws/install/setup.bash

# Start the ROS2 launch file
ros2 launch reachy_kinematics kinematics.launch.py