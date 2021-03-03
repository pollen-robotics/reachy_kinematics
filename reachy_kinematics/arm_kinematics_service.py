from functools import partial
from typing import List

import numpy as np

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from reachy_msgs.srv import GetArmFK, GetArmIK

from .kinematics import generate_solver, forward_kinematics, inverse_kinematics


class ArmKinematicsService(Node):
    def __init__(self) -> None:
        super().__init__('arm_kinematics_service')
        self.logger = self.get_logger()

        self.retrieve_urdf()
        chains, fk_solvers, ik_solvers = generate_solver(self.urdf)

        for side in ('left', 'right'):
            srv = self.create_service(
                srv_type=GetArmFK,
                srv_name=f'/{side}_arm/kinematics/forward',
                callback=partial(self.arm_fk, side=side, solver=fk_solvers[side], nb_joints=chains[side].getNrOfJoints()),
            )
            self.logger.info(f'Starting service "{srv.srv_name}".')

            srv = self.create_service(
                srv_type=GetArmIK,
                srv_name=f'/{side}_arm/kinematics/inverse',
                callback=partial(self.arm_ik, side=side, solver=ik_solvers[side], nb_joints=chains[side].getNrOfJoints()),
            )
            self.logger.info(f'Starting service "{srv.srv_name}".')

        self.logger.info('Node ready!')

    def arm_fk(self, request: GetArmFK.Request, response: GetArmFK.Response, side: str, solver, nb_joints: int) -> GetArmFK.Response:
        joints = self.joint_state_as_list(request.joint_position, side)
        if len(joints) != nb_joints:
            return response

        res, M = forward_kinematics(solver, request.joint_position.position, nb_joints)
        q = Rotation.from_matrix(M[:3, :3]).as_quat()

        response.success = True
        response.pose.position = Point(x=M[0, 3], y=M[1, 3], z=M[2, 3])
        response.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return response

    def arm_ik(self, request: GetArmIK.Request, response: GetArmIK.Response, side: str, solver, nb_joints: int) -> GetArmIK.Response:
        if request.q0.position:
            q0 = self.joint_state_as_list(request.q0, side)

            if len(q0) != nb_joints:
                self.logger.warning(f'Wrong number of joints provided ({len(q0)} instead of {nb_joints})!')
                return response
        else:
            q0 = self.get_default_q0(side)

        M = np.eye(4)
        p = request.pose.position
        M[:3, 3] = p.x, p.y, p.z
        q = request.pose.orientation
        M[:3, :3] = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()

        res, J = inverse_kinematics(solver, q0, M, nb_joints)

        response.joint_position.name = self.get_arm_joints_name(side)
        response.joint_position.position = list(J)

        return response

    def get_arm_joints_name(self, side: str) -> List[str]:
        side = 'l' if side == 'left' else 'r'

        return [
            f'{side}_shoulder_pitch',
            f'{side}_shoulder_roll',
            f'{side}_arm_yaw',
            f'{side}_elbow_pitch',
            f'{side}_forearm_yaw',
            f'{side}_wrist_pitch',
            f'{side}_wrist_roll',
        ]

    def get_default_q0(self, side: str) -> List[float]:
        return [0, 0, 0, -np.pi / 2, 0, 0, 0]

    def joint_state_as_list(self, joint_state: JointState, side: str) -> List[float]:
        joint_names = self.get_arm_joints_name(side)
        if len(joint_state.position) != len(joint_names):
            self.logger.warning(f'Wrong number of joints provided ({len(joint_state.position)} instead of {len(joint_names)})!')
            raise ValueError

        if joint_state.name:
            positions = [0.0] * len(joint_names)

            for name, pos in zip(joint_state.name, joint_state.position):
                positions[joint_names.index(name)] = pos
            return positions

        else:
            return joint_state.position

    def retrieve_urdf(self, timeout_sec: float = 5) -> None:
        self.logger.info('Retrieving URDF from "/robot_description"...')

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
            self.logger.error('Could not retrieve the URDF!')
            raise EnvironmentError('Could not retrieve the URDF!')


def main(args=None):
    rclpy.init(args=args)

    arm_kinematics_service = ArmKinematicsService()
    rclpy.spin(arm_kinematics_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
