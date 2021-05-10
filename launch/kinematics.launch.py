import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    urdf = os.path.join(
        get_package_share_directory('reachy_description'),
        'urdf/reachy.URDF',
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            arguments=[urdf],
        ),
        Node(
            package='reachy_kinematics',
            executable='orbita_kinematics_service',
        ),
        Node(
            package='reachy_kinematics',
            executable='arm_kinematics_service',
        ),
    ])
