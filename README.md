# Reachy kinematics

ROS2 package publishing the /robot_description of Reachy using its URDF and creating the services handling the kinematics of both arms and Orbita actuator.


**ROS2 Version: Foxy**

Dependencies: [reachy_msgs](https://github.com/pollen-robotics/reachy_msgs)

How to install:

```bash
cd ~/reachy_ws/src
git clone https://github.com/pollen-robotics/reachy_kinematics.git
cd ~/reachy_ws/
colcon build --packages-select reachy_kinematics
```

To install PyKDL (on Ubuntu 20.04):
```bash
sudo apt install python3-pykdl
```

## Published topics

* **/robot_description**  ([std_msgs/msg/String](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/String.html)) - Description of Reachy's URDF as a string. Needed by the arm kinematics computation. </br> More information can be found on the [robot_state_publisher GitHub](https://github.com/ros/robot_state_publisher/tree/foxy).


## Services

* **/left_arm/kinematics/forward** ([reachy_msgs/srv/GetArmFK.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetArmFK.srv)) - Compute the forward kinematics for the given positions of left arm motors. Returns the pose in the robot frame of the joint at the end of the requested kinematic chain.

* **/left_arm/kinematics/inverse** ([reachy_msgs/srv/GetArmIK.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetArmIK.srv)) - Compute the inverse kinematics for the requested joint pose in a left arm kinematic chain. Returns the joints goal positions to reachy the joint pose.

* **/right_arm/kinematics/forward** ([reachy_msgs/srv/GetArmFK.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetArmFK.srv)) - Compute the forward kinematics for the given positions of right arm motors. Returns the pose in the robot frame of the joint at the end of the requested kinematic chain.

* **/right_arm/kinematics/inverse** ([reachy_msgs/srv/GetArmIK.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetArmIK.srv)) - Compute the inverse kinematics for the requested joint pose in a right arm kinematic chain. Returns the joints goal positions to reachy the joint pose.

* **/orbita/kinematics/inverse** ([reachy_msgs/srv/GetOrbitaIK.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetOrbitaIK.srv)) - Compute the inverse kinematics for the Orbita actuator. Returns the disks goal positions for the requested quaternion.

* **/orbita/kinematics/look_vector_to_quaternion** ([reachy_msgs/srv/GetQuaternionTransform.srv](https://github.com/pollen-robotics/reachy_msgs/blob/master/srv/GetQuaternionTransform.srv)) - Compute the quaternion corresponding to the target point in the look at instruction. 

## Launch files

* **description.launch.py** - Publish Reachy's urdf in /robot_description, needed by the arm kinematics solver.
* **arm_kinematics.launch.py** - Launch the two kinematics services for Reachy's arms.
* **orbita_kinematics_service.launch.py** - Launch the two kinematics services for Orbita.
* **kinematics.launch.py** - Start the three launch files at once.

---
This package is part of the ROS2-based software release of the version 2021 of Reachy.


Visit [pollen-robotics.com](https://pollen-robotics.com) to learn more or visit [our forum](https://forum.pollen-robotics.com) if you have any questions.
