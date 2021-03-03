'''Doc.'''
import rclpy
from rclpy.node import Node

from reachy_msgs.srv import GetOrbitaIK, GetQuaternionTransform

from .orbita_kinematics import OrbitaKinSolver


class OrbitaKinematicsService(Node):
    def __init__(self):
        super().__init__('orbita_kinematics_service')
        self.logger = self.get_logger()
        self.kin_solver = OrbitaKinSolver()
        self.ik_service = self.create_service(GetOrbitaIK, 'orbita/kinematics/inverse', self.ik_callback)
        self.logger.info(f'Starting service "{self.ik_service.srv_name}".')
        self.orbita_look_at_tf_service = self.create_service(GetQuaternionTransform, 'orbita/kinematics/look_vector_to_quaternion', self.vec2quat_callback)
        self.logger.info(f'Starting service "{self.orbita_look_at_tf_service.srv_name}".')

        self.logger.info('Node ready!')


    def ik_callback(self, request, response):
        '''
        request: 
            - geometry_msgs/Quaternion quat
        response:
            - sensor_msgs/JointState disk_pos
        '''
        try:
            response.disk_pos = self.kin_solver.orbita_ik(request.quat)
            response.success = True
        except ValueError:
            self.logger.warning(f'Math domain error after ik request: {request.quat}')
            response.success = False
        return response

    def vec2quat_callback(self, request, response):
        response.quat = self.kin_solver.find_quaternion_transform(request.point)
        return response


def main(args=None):
    rclpy.init(args=args)

    orb_kin_service = OrbitaKinematicsService()

    rclpy.spin(orb_kin_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()