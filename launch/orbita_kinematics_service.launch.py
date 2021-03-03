from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='reachy_kinematics',
            executable='orbita_kinematics_service',
        ),
    ])
