import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    urdf = os.path.join(
        get_package_share_directory('reachy_kinematics'),
        'reachy.URDF',
    )

    rviz_config = os.path.join(
        get_package_share_directory('reachy_kinematics'),
        'reachy.rviz',
    )

    return LaunchDescription([
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            parameters=[
                {'-d', rviz_config},
            ],
        ),
    ])
