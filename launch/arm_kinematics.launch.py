from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='reachy_kinematics',
            executable='arm_kinematics_service',
        ),
    ])
