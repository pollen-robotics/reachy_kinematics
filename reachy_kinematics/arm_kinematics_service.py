"""ROS2 Foxy package for Reachy arm kinematics.

See README.md for details on the services exposed.

"""
from functools import partial
from typing import List

import numpy as np

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
import PyKDL as kdl

from geometry_msgs.msg import Point, Quaternion, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from reachy_msgs.srv import GetArmFK, GetArmIK

from .arm_kinematics import generate_solver, forward_kinematics, inverse_kinematics, get_jacobian, jacobian_pseudo_inverse, orientation_difference
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
        chains, fk_solvers, ik_solvers, jac_solvers, self.kdl_tree = generate_solver(
            self.urdf)

        self.urdf_model=urdf.URDF.from_xml_string(self.urdf)
        self.upper_limits={}
        self.lower_limits={}
        self.lower_singularity_threshold=25.0
        self.hard_stop_singularity_threshold=40.0
        self.max_joints_vel=1.0*(1.0/50.0) #max delta_q assuming 1 rad/s and a 50Hz update... 
        
        #self.logger.warning('urdf: {}'.format(self.urdf))        
        for side in ('left', 'right'):
            srv = self.create_service(
                srv_type=GetArmFK,
                srv_name=f'/{side}_arm/kinematics/forward',
                callback=partial(
                    self.arm_fk, side=side, solver=fk_solvers[side], nb_joints=chains[side].getNrOfJoints()),
            )
            self.logger.info(f'Starting service "{srv.srv_name}".')

            srv = self.create_service(
                srv_type=GetArmIK,
                srv_name=f'/{side}_arm/kinematics/inverse',
                callback=partial(
                    self.arm_ik, side=side, solver=ik_solvers[side], nb_joints=chains[side].getNrOfJoints()),
            )
            self.logger.info(f'Starting service "{srv.srv_name}".')


            srv = self.create_service(
                srv_type=GetArmIK,
                srv_name=f'/{side}_arm/kinematics/safer_inverse',
                callback=partial(
                    self.arm_ik_safe, side=side, fk_solver=fk_solvers[side], jac_solver=jac_solvers[side], nb_joints=chains[side].getNrOfJoints()),
            )
            self.logger.info(f'Starting service "{srv.srv_name}".')

            
            sub = self.create_subscription(
                Pose,
                f'/{side}_arm/servo_goals',
                partial(
                    self.arm_servo, side=side, fk_solver=fk_solvers[side], jac_solver=jac_solvers[side]),
                10)

            up,lo=self.get_limits(self.urdf_model,side)
            self.upper_limits[side]=up
            self.lower_limits[side]=lo

        sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_cb,
            10)

        self.goal_publisher = self.create_publisher(
            JointState, 'joint_goals', 1)
        self.current_joint_states = None


        self.logger.warning('limits: {} {}'.format(self.upper_limits, self.lower_limits))
        
        self.logger.info('Node ready!')


    def get_chain_joint_names(self, end_link, links=False, fixed=False):
         return self.urdf_model.get_chain('torso', end_link,
                                    links=links, fixed=fixed)
        
    def get_limits(self, urdf, side):

        # adapted from pykdl_utils
        # record joint information in easy-to-use lists
        joint_limits_lower = []
        joint_limits_upper = []

        joint_types = []
        chain=self.get_chain_joint_names(f'{side}_tip')
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
        
        joint_limits_lower = np.array([replace_none(jl, -np.inf) 
                                            for jl in joint_limits_lower])
        joint_limits_upper = np.array([replace_none(jl, np.inf) 
                                            for jl in joint_limits_upper])

        return joint_limits_lower, joint_limits_upper

    def check_joints_in_limits(self, q, side):

        lower_lim=np.minimum(self.lower_limits[side],self.upper_limits[side])
        upper_lim=np.maximum(self.lower_limits[side],self.upper_limits[side])
                
        return np.all([q >= lower_lim, q <= upper_lim], 0)


    def clip_joints_limits(self, q, side):
        
        lower_lim=np.minimum(self.lower_limits[side],self.upper_limits[side])
        upper_lim=np.maximum(self.lower_limits[side],self.upper_limits[side])

        return np.clip(q, lower_lim, upper_lim)

    def check_joints_in_vel_limits(self, dq):
        return np.all([dq >= -self.max_joints_vel, dq <= self.max_joints_vel], 0)


    def clip_joints_vel_limits(self, dq):
        return np.clip(dq, -self.max_joints_vel, self.max_joints_vel)
    
    
    def joint_states_cb(self, states: JointState) -> None:
        self.current_joint_states = states


    def scaling_factor_for_singularity(self, delta_x_in, u,s,vh, Jinv, joint_states, jac_solver):
        """
        Adapted from Moveit Servo
        Try to scale the command when going close to singularity.
        """
        vel_scale=1.0
        ndim=len(delta_x_in)

        # Find the direction away from nearest singularity.
        # The last column of U from the SVD of the Jacobian points directly toward or away from the singularity.
        # The sign can flip at any time, so we have to do some extra checking.
        # Look ahead to see if the Jacobian's condition will decrease.
        vector_toward_singularity = np.array([i for i in u[:,ndim-1]]).flatten()

        ini_condition = s[0] / s[len(s)-1]
        
        # This singular vector tends to flip direction unpredictably. See R. Bro,
        # "Resolving the Sign Ambiguity in the Singular Value Decomposition".
        # Look ahead to see if the Jacobian's condition will decrease in this
        # direction. Start with a scaled version of the singular vector
        delta_x = np.zeros(ndim)
        scale = 100.0
        delta_x = vector_toward_singularity / scale

        # Calculate a small change in joints
        
        delta_theta = Jinv @ delta_x
        new_theta=np.array(joint_states)+delta_theta

        newJ = get_jacobian(joint_states, jac_solver)        
        
        newu, news, newvh=np.linalg.svd(newJ, full_matrices=False)
        new_condition = news[0] / news[len(news)-1]
        
        # If new_condition < ini_condition, the singular vector does point towards a
        # singularity. Otherwise, flip its direction.
        if ini_condition >= new_condition:
            vector_toward_singularity *= -1;

        #If this dot product is positive, we're moving toward singularity ==> decelerate
        dot=np.dot(vector_toward_singularity, delta_x_in)
        
        if dot > 0:
            # Ramp velocity down linearly when the Jacobian condition is between lower_singularity_threshold and
            # hard_stop_singularity_threshold, and we're moving towards the singularity
            if (ini_condition > self.lower_singularity_threshold) and (ini_condition < self.hard_stop_singularity_threshold):
                
                vel_scale = 1.0 - (ini_condition - self.lower_singularity_threshold) / (self.hard_stop_singularity_threshold - self.lower_singularity_threshold)

                self.logger.warning("We decelerate for singularity ({})".format(ini_condition))
                
            # Very close to singularity, so halt.
            elif ini_condition > self.hard_stop_singularity_threshold:
                vel_scale = 0;
                self.logger.warning("We stop for singularity ({})".format(ini_condition))
                
                
        return vel_scale          


    def get_current_state(self, side, fk_solver):
        
        try:
            j = JointState()

            for name, pos in zip(self.current_joint_states.name, self.current_joint_states.position):
                if 'gripper' not in name:
                    j.name.append(name)
                    j.position.append(pos)
            joints = self._joint_state_as_list(j, side)
        except ValueError:
            self.logger.error('Bad joint states')
            return None, None


        
        res, M = forward_kinematics(
            fk_solver, joints, len(joints))
        #euler = Rotation.from_matrix(M[:3, :3]).as_euler('xyz')

        R = M[:3, :3].flatten().tolist()

        #current pose
        pose0 = kdl.Frame()
        pose0.p = kdl.Vector(M[0, 3], M[1, 3], M[2, 3])
        pose0.M = kdl.Rotation(*R)

        return joints, pose0


    def safe_ik(self, cmd, delta_x, joints, side,jac_solver):
    
        delta_x_norm=np.linalg.norm(delta_x)
        if delta_x_norm>0.0001:
            #TODO check validity of goal and current state. Handle the no orientation goal case
            J = get_jacobian(joints, jac_solver)

            #pseudo inverse Jacobian through SVD decomposition
            u, s, vh=np.linalg.svd(J, full_matrices=False)
            invs=np.linalg.inv(np.diag(s))
            Jinv_svd=np.dot(vh.transpose(),np.dot(np.diag(s**-1),u.transpose()))

            delta_q=Jinv_svd @ delta_x 

            # apply a scaling for the command
            delta_q*=self.scaling_factor_for_singularity(delta_x, u,s,vh, Jinv_svd, joints, jac_solver)
            delta_q=np.array([delta_q.flat[i] for i,_ in enumerate(cmd.position)]) #why the f*ck I cannot flatten this one?

            #check the result velocity
            if not self.check_joints_in_vel_limits(delta_q).any():
                self.logger.warning("Trying to move outside joints vel limits! {}".format(delta_q))
                delta_q=self.clip_joints_vel_limits(delta_q)
                self.logger.warning("\tclipped vel {}".format(delta_q))
            newpos=np.zeros(len(cmd.position))
            for i,_ in enumerate(cmd.position):
                
                newpos[i]=cmd.position[i]+delta_q[i]

            # check the angle limits 
            if not self.check_joints_in_limits(cmd.position, side).any():
                self.logger.warning("Trying to move outside joints limits! {} ({} {})".format(newpos,self.lower_limits[side], self.upper_limits[side]))
                newpos=self.clip_joints_limits(newpos,side)

            cmd.position=[p for p in newpos]
            return cmd

        else:
            return cmd
                
    def arm_servo(self, goal: Pose, side: str, fk_solver, jac_solver) -> None:
        """
        Compute a goal joint position from a goal cartesian position and the current joint state using the inverse Jacobian (differential kinematics).
        Trying to implement something roughly similar to cartesianServoCalcs from Moveit2
        \delta q = J^{-1} \delta x
        """

        # TODO check if the commands are valid (NaN, inf, non normalized quaternion...)

        # the cartesian goal is in the robot frame

        # compute delta_x (delta of cartesian pos):
        # First, get the current cartesian pos from joints state



        joints, pose0 = self.get_current_state(side, fk_solver)
        if joints is None or pose0 is None:
            self.logger.error('Aborting')
            return

        #copy the current joint state
        cmd=JointState()
        for name, pos in zip(self.current_joint_states.name, self.current_joint_states.position):
            if 'gripper' not in name:
                cmd.name.append(name)
                cmd.position.append(pos)
        
        #goal pos
        pose1 = kdl.Frame()
        pose1.p = kdl.Vector(goal.position.x,goal.position.y,goal.position.z)
        R1 = Rotation.from_quat((goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w)).as_matrix()
        R1=R1.flatten().tolist()
        pose1.M=kdl.Rotation(*R1)

        #self.logger.info("SERVO pose0: {} pose1: {}".format(pose0,pose1))
        #delta of the pose
        delta_x=kdl.diff(pose0,pose1,1)
        #self.logger.info("SERVO kdl delta_x: {}".format(delta_x))
                
        delta_x=np.array([delta_x.vel.x(),delta_x.vel.y(),delta_x.vel.z(),delta_x.rot.x(),delta_x.rot.y(),delta_x.rot.z()])
        #self.logger.info("SERVO kdl delta_x np: {}".format(delta_x))

        cmd=self.safe_ik(cmd,delta_x,joints,side,jac_solver)

        self.goal_publisher.publish(cmd)
            
        
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

        res, M = forward_kinematics(
            solver, request.joint_position.position, nb_joints)
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

        M = np.eye(4)
        p = request.pose.position
        M[:3, 3] = p.x, p.y, p.z
        q = request.pose.orientation
        M[:3, :3] = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()

        res, J = inverse_kinematics(solver, q0, M, nb_joints)

        response.success = True
        response.joint_position.name = self.get_arm_joints_name(side)
        response.joint_position.position = list(J)

        return response



    def arm_ik_safe(self, request: GetArmIK.Request, response: GetArmIK.Response, side: str, fk_solver, jac_solver, nb_joints: int) -> GetArmIK.Response:
        """Compute the inverse arm kinematics given the request."""

        # #get current joints state

        joints, pose0 = self.get_current_state(side, fk_solver)
        if joints is None or pose0 is None:
            self.logger.error('Aborting')
            return

        #copy the current joint state
        cmd=JointState()
        for name, pos in zip(self.current_joint_states.name, self.current_joint_states.position):
            if 'gripper' not in name:
                cmd.name.append(name)
                cmd.position.append(pos)
        

        #goal pos
        pose1 = kdl.Frame()
        pose1.p = kdl.Vector(request.pose.position.x,request.pose.position.y,request.pose.position.z)
        R1 = Rotation.from_quat((request.pose.orientation.x, request.pose.orientation.y, request.pose.orientation.z, request.pose.orientation.w)).as_matrix()
        R1=R1.flatten().tolist()
        pose1.M=kdl.Rotation(*R1)

        #delta of the pose
        delta_x=kdl.diff(pose0,pose1,1)
                
        delta_x=np.array([delta_x.vel.x(),delta_x.vel.y(),delta_x.vel.z(),delta_x.rot.x(),delta_x.rot.y(),delta_x.rot.z()])
        

        cmd=self.safe_ik(cmd,delta_x,joints,side,jac_solver)
        self.logger.info("SAFE IK : {}".format(cmd))        
        response.success = True
        response.joint_position=cmd

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
