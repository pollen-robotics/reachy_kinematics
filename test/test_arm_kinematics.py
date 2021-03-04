import os
from pathlib import Path

import pytest

import numpy as np
import PyKDL as kdl
from reachy_kinematics.kdl_parser_py import urdf
from reachy_kinematics import arm_kinematics

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from std_msgs.msg import String

def urdf_retriever():
    class UrdfNodeRetriever(Node):
        def retrieve_urdf(self, timeout_sec: float = 5) -> None:
            """Retrieve the URDF file from the /robot_description topic.

            Will raise an EnvironmentError if the topic is unavailable.
            """
            qos_profile = QoSProfile(depth=1)
            qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

            self.urdf = None

            def urdf_received(msg: String):
                self.urdf = msg.data

            self.create_subscription(
                msg_type=String, topic='/robot_description',
                qos_profile=qos_profile,
                callback=urdf_received,
            )
            rclpy.spin_once(self, timeout_sec=timeout_sec)
            if self.urdf is None:
                raise EnvironmentError('Could not retrieve the URDF!')

    rclpy.init()
    node = UrdfNodeRetriever('urdf_retriever')
    node.retrieve_urdf()
    urdf = node.urdf
    rclpy.shutdown()
    return urdf


class TestArmKinematics():
    @classmethod
    def setup_class(cls):
        cls.urdf = urdf_retriever()
        _, cls.fk_solvers, cls.ik_solvers = arm_kinematics.generate_solver(cls.urdf)

    def test_forward_kinematics_right(self):
        joints = np.array([0, 0, 0, 0, 0, 0, 0])
        nb_joints = 7
        _, M_right = arm_kinematics.forward_kinematics(self.fk_solvers['right'], joints, nb_joints)
        assert np.array_equal(M_right, np.array((
            (1, 0, 0, 0),
            (0, 1, 0, -0.202),
            (0, 0, 1, -0.6475),
            (0, 0, 0, 1),
        )))

    def test_forward_kinematics_left(self):
        joints = np.array([0, 0, 0, 0, 0, 0, 0])
        nb_joints = 7
        _, M_left = arm_kinematics.forward_kinematics(self.fk_solvers['left'], joints, nb_joints)
        assert np.array_equal(M_left, np.array((
            (1, 0, 0, 0),
            (0, 1, 0, 0.202),
            (0, 0, 1, -0.6475),
            (0, 0, 0, 1),
        )))

    def test_inverse_kinematics_right(self):
        q0 = np.array([0, 0, 0, -np.pi/2, 0, 0, 0])
        target_pose = np.array((
            (1, 0, 0, 0),
            (0, 1, 0, -0.202),
            (0, 0, 1, -0.6475),
            (0, 0, 0, 1),
        ))
        nb_joints = 7
        _, sol = arm_kinematics.inverse_kinematics(self.ik_solvers['right'], q0, target_pose, nb_joints)
        assert np.sum(sol) < 0.01

    def test_inverse_kinematics_left(self):
        q0 = np.array([0, 0, 0, np.pi/2, 0, 0, 0])
        target_pose = np.array((
            (1, 0, 0, 0),
            (0, 1, 0, 0.202),
            (0, 0, 1, -0.6475),
            (0, 0, 0, 1),
        ))
        nb_joints = 7
        _, sol = arm_kinematics.inverse_kinematics(self.ik_solvers['left'], q0, target_pose, nb_joints)
        assert np.sum(sol) < 0.01
