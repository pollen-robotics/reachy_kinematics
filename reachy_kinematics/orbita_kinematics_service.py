'''Doc.'''
import rclpy
from rclpy.node import Node

from reachy_msgs.srv import GetOrbitaIK, GetQuaternionTransform

from .orbita_kinematics import OrbitaKinSolver


class OrbitaKinService(Node):
    def __init__(self):
        super().__init__('orbita_kinematics_service')
        self.kin_solver = OrbitaKinSolver()
        self.ik_service = self.create_service(GetOrbitaIK, 'orbita_ik', self.ik_callback)
        self.orbita_look_at_tf_service = self.create_service(GetQuaternionTransform, 'orbita_look_at_tf', self.quat_tf_callback)
        self.logger = self.get_logger()
        self.logger.info('Initialized OrbitaKiService.')

    def ik_callback(self, request, response):
        '''
        request: 
            - geometry_msgs/Quaternion quat
        response:
            - sensor_msgs/JointState disk_pos
        '''
        self.logger.debug(f'Received request ik: {request.quat}')
        try:
            response.disk_pos = self.kin_solver.orbita_ik(request.quat)
            response.success = True
        except ValueError:
            self.logger.warning(f'Math domain error after ik request: {request.quat}')
            response.success = False
        return response

    def quat_tf_callback(self, request, response):
        response.quat = self.kin_solver.find_quaternion_transform(request.point)
        return response


def main(args=None):
    rclpy.init(args=args)

    orb_kin_service = OrbitaKinService()

    rclpy.spin(orb_kin_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()