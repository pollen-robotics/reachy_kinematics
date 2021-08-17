"""ROS2 Foxy package for Reachy arm kinematics.

See README.md for details on the services exposed.

"""
from functools import partial
from typing import List, Tuple

import numpy as np

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from reachy_msgs.srv import GetArmFK, GetArmIK

from .arm_kinematics import generate_solver, forward_kinematics, inverse_kinematics
from .urdf_parser_py import urdf


class ArmKinematicsService(Node):
    """Node exposing the reachy arm kinematics services.

    Services:
    - /left_arm/kinematics/forward reachy_msgs/srv/GetArmFK.srv
    - /left_arm/kinematics/inverse reachy_msgs/srv/GetArmIK.srv
    - /right_arm/kinematics/forward reachy_msgs/srv/GetArmFK.srv
    - /right_arm/kinematics/inverse reachy_msgs/srv/GetArmFK.srv
    """

    def __init__(self) -> None:
        """Set up the node."""
        super().__init__('arm_kinematics_service')
        self.logger = self.get_logger()

        self.retrieve_urdf()
        chains, fk_solvers, ik_solvers = generate_solver(self.urdf)

        self.urdf_model = urdf.URDF.from_xml_string(self.urdf)
        self.upper_limits = {}
        self.lower_limits = {}
        self.max_joints_vel = 3.0 * (1.0 / 50.0)  # max delta_q assuming 1 rad/s and a 50Hz update...

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

            up, lo = self.get_limits(self.urdf_model, side)
            self.upper_limits[side] = up
            self.lower_limits[side] = lo

        self.logger.info('Found angle limits: {} {}'.format(self.upper_limits, self.lower_limits))

        self.current_joint_states = None
        sub = self.create_subscription(
            msg_type=JointState,
            topic='joint_states',
            callback=self.joint_states_cb,
            qos_profile=5,
        )
        self.logger.info(f'Subscribe to topic "{sub.topic_name}".')

        self.logger.info('Node ready!')

    def get_chain_joint_names(self, end_link: str, links=False, fixed=False):
        """Get name of each joints from torso to specfied end link."""
        return self.urdf_model.get_chain('torso', end_link, links=links, fixed=fixed)

    def get_limits(self, urdf, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get all joints limits from the urdf."""
        # adapted from pykdl_utils
        # record joint information in easy-to-use lists
        joint_limits_lower = []
        joint_limits_upper = []

        joint_types = []
        chain = self.get_chain_joint_names(f'{side}_tip')
        self.logger.warning('chain: {}'.format(chain))
        for jnt_name in chain:
            jnt = urdf.joint_map[jnt_name]
            if jnt.limit is not None:
                joint_limits_lower.append(jnt.limit.lower)
                joint_limits_upper.append(jnt.limit.upper)
            else:
                joint_limits_lower.append(None)
                joint_limits_upper.append(None)

            joint_types.append(jnt.joint_type)

        def replace_none(x, v):
            if x is None:
                return v
            return x

        joint_limits_lower = np.array([replace_none(jl, -np.inf) for jl in joint_limits_lower])
        joint_limits_upper = np.array([replace_none(jl, np.inf) for jl in joint_limits_upper])

        return joint_limits_lower, joint_limits_upper

    def check_joints_in_limits(self, q: np.ndarray, side: str) -> np.ndarray:
        """Check if the given solution is within the joints limits."""
        lower_lim = np.minimum(self.lower_limits[side], self.upper_limits[side])
        upper_lim = np.maximum(self.lower_limits[side], self.upper_limits[side])

        return np.all([q >= lower_lim, q <= upper_lim], 0)

    def clip_joints_limits(self, q: np.ndarray, side: str) -> np.ndarray:
        """Clip the given solution within the joints limits."""
        lower_lim = np.minimum(self.lower_limits[side], self.upper_limits[side])
        upper_lim = np.maximum(self.lower_limits[side], self.upper_limits[side])

        return np.clip(q, lower_lim, upper_lim)

    def check_joints_in_vel_limits(self, dq: np.ndarray) -> np.ndarray:
        """Check if the given velocity is within the authorized limits."""
        return np.all([dq >= -self.max_joints_vel, dq <= self.max_joints_vel], 0)

    def clip_joints_vel_limits(self, dq: np.ndarray) -> np.ndarray:
        """Clip the given velocity solution  within the velocity limits."""
        return np.clip(dq, -self.max_joints_vel, self.max_joints_vel)

    def joint_states_cb(self, states: JointState) -> None:
        """Get latest joint states."""
        self.current_joint_states = states

    def arm_fk(self, request: GetArmFK.Request, response: GetArmFK.Response, side: str, solver, nb_joints: int) -> GetArmFK.Response:
        """Compute the forward arm kinematics given the request."""
        try:
            joints = self._joint_state_as_list(request.joint_position, side)
        except ValueError:
            response.success = False
            return response

        if len(joints) != nb_joints:
            response.success = False
            return response

        res, M = forward_kinematics(solver, request.joint_position.position, nb_joints)
        q = Rotation.from_matrix(M[:3, :3]).as_quat()

        response.success = True
        response.pose.position = Point(x=M[0, 3], y=M[1, 3], z=M[2, 3])
        response.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return response

    def arm_ik(self, request: GetArmIK.Request, response: GetArmIK.Response, side: str, solver, nb_joints: int) -> GetArmIK.Response:
        """Compute the inverse arm kinematics given the request."""
        if request.q0.position:
            try:
                q0 = self._joint_state_as_list(request.q0, side)
            except ValueError:
                response.success = False
                return response

            if len(q0) != nb_joints:
                self.logger.warning(f'Wrong number of joints provided ({len(q0)} instead of {nb_joints})!')
                response.success = False
                return response
        else:
            q0 = self.get_default_q0(side)

        # Get current joints state
        j = JointState()
        for name, pos in zip(self.current_joint_states.name, self.current_joint_states.position):
            if 'gripper' not in name and name in self.get_arm_joints_name(side):
                j.name.append(name)
                j.position.append(pos)

        joints = self._joint_state_as_list(j, side)

        M = np.eye(4)
        p = request.pose.position
        M[:3, 3] = p.x, p.y, p.z
        q = request.pose.orientation
        M[:3, :3] = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()

        res, J = inverse_kinematics(solver, q0, M, nb_joints)

        delta_q = np.array(J) - np.array(joints)

        # Check the result velocity
        if not self.check_joints_in_vel_limits(delta_q).any():
            self.logger.warning("Trying to move outside joints vel limits! {}".format(delta_q))
            delta_q = self.clip_joints_vel_limits(delta_q)
            self.logger.warning("\tclipped vel {}".format(delta_q))

        joints_goal = np.array(joints) + delta_q

        # check the angle limits
        if not self.check_joints_in_limits(joints_goal, side).any():
            self.logger.warning("Trying to move outside joints limits! {} ({} {})".format(joints_goal, self.lower_limits[side], self.upper_limits[side]))
            joints_goal = self.clip_joints_limits(joints_goal, side)

        response.success = True
        response.joint_position.name = self.get_arm_joints_name(side)
        response.joint_position.position = list(joints_goal)

        return response

    def get_arm_joints_name(self, side: str) -> List[str]:
        """Return the list of joints name for the specified arm."""
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
        """Return the default position value for the joints of the specified arm.

        The default position corresponds to Reachy with a straight arm and the elbow flexed at 90 degrees.
        """
        return [0, 0, 0, -np.pi / 2, 0, 0, 0]

    def _joint_state_as_list(self, joint_state: JointState, side: str) -> List[float]:
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
        """Retrieve the URDF file from the /robot_description topic.

        Will raise an EnvironmentError if the topic is unavailable.
        """
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
    """Run main entry point."""
    rclpy.init(args=args)

    arm_kinematics_service = ArmKinematicsService()
    rclpy.spin(arm_kinematics_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
