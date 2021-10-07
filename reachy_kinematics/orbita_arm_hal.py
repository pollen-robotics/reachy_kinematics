#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: orbita_arm_hal.py
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: Friday, September 24 2021
#  Revised	:
#  Version	:
#  Target MCU	:
#
#  This code is distributed under the GNU Public License
# 		which can be found at http://www.gnu.org/licenses/gpl.txt
#
#
#  Notes:	notes
#


"""ROS2 Foxy package for Reachy orbita kinematics.

See README.md for details on the services exposed.

"""
import rclpy
from rclpy.node import Node

# from .orbita_arm_kinematics import OrbitaArmKinematicsSolver
from reachy_kinematics.orbita_arm_kinematics import OrbitaArmKinematicsSolver

from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectory

from copy import copy
import tf_transformations
from geometry_msgs.msg import Quaternion, Point


rd_names = [
    'painteffector_disk_top, painteffector_disk_middle, painteffector_disk_bottom']


r_dummy = ['r_dummy_wrist_roll', 'r_dummy_wrist_pitch', 'r_dummy_wrist_yaw']


class OrbitaArmHAL(Node):

    def __init__(self):
        """Set up the node."""
        super().__init__('orbita_arm_hal')
        self.logger = self.get_logger()
        self.kin_solver_arm = OrbitaArmKinematicsSolver(rd_names)
        self.logger.info('Node ready!')

        self.cmd_publisher = self.create_publisher(
            JointState, 'joint_goals', 1)
        self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_cb,
            10)

        self.create_subscription(
            JointTrajectory,
            '/reachy_left_arm_controller/joint_trajectory',
            self.left_arm_cb,
            10)

        self.create_subscription(
            JointTrajectory,
            '/reachy_right_arm_controller/joint_trajectory',
            self.right_arm_cb,
            10)

        self.current_state = None

    def send_joint(self, msg):

        new = copy(self.current_state)

        for joint in msg.joint_names:
            if joint not in r_dummy:
                p = msg.points[0].positions[msg.joint_names.index(joint)]
                new.position[new.name.index(joint)] = p

        self.cmd_publisher.publish(new)

    def send_orbita(self, msg):

        roll = msg.points[0].positions[msg.joint_names.index(
            'r_dummy_wrist_roll')]
        pitch = msg.points[0].positions[msg.joint_names.index(
            'r_dummy_wrist_pitch')]
        yaw = msg.points[0].positions[msg.joint_names.index(
            'r_dummy_wrist_yaw')]

        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)

        quat = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        # el = tf_transformations.euler_from_quaternion(q)
        # print("orig: {}".format((roll, pitch, yaw)))
        # print("euler: {}".format(el))
        # print("quat: {}".format(quat))
        try:
            r = self.kin_solver_arm.orbita_arm_ik(quat)
            # print("disks: {}".format(r))
            self.cmd_publisher.publish(r)
        except:
            self.get_logger().warn('Orbita ik error!')

    def joint_states_cb(self, msg):
        self.current_state = msg

    def left_arm_cb(self, msg):
        self.send_joint(msg)
        self.send_orbita(msg)

    def right_arm_cb(self, msg):
        self.send_joint(msg)
        self.send_orbita(msg)


def main(args=None):
    rclpy.init(args=args)

    hal = OrbitaArmHAL()

    rclpy.spin(hal)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
