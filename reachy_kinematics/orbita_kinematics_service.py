"""ROS2 Foxy package for Reachy orbita kinematics.

See README.md for details on the services exposed.

"""
import rclpy
from rclpy.node import Node

from reachy_msgs.srv import GetOrbitaIK, GetQuaternionTransform

from .orbita_kinematics import OrbitaKinSolver


class OrbitaKinematicsService(Node):
    """Node exposing the reachy orbita kinematics services."""

    def __init__(self):
        """Set up the node."""
        super().__init__('orbita_kinematics_service')
        self.logger = self.get_logger()
        self.kin_solver = OrbitaKinSolver()
        self.ik_service = self.create_service(GetOrbitaIK, 'orbita/kinematics/inverse', self.ik_callback)
        self.logger.info(f'Starting service "{self.ik_service.srv_name}".')
        self.orbita_look_at_tf_service = self.create_service(GetQuaternionTransform, 'orbita/kinematics/look_vector_to_quaternion', self.vec2quat_callback)
        self.logger.info(f'Starting service "{self.orbita_look_at_tf_service.srv_name}".')

        self.logger.info('Node ready!')

    def ik_callback(self, request: GetOrbitaIK.Request, response: GetOrbitaIK.Response):
        """Compute the inverse kinematics for orbita given the request."""
        try:
            response.disk_pos = self.kin_solver.orbita_ik(request.quat)
            response.success = True
        except ValueError:
            self.logger.warning(f'Math domain error after ik request: {request.quat}')
            response.success = False
        return response

    def vec2quat_callback(self, request: GetQuaternionTransform.Request, response: GetQuaternionTransform.Response):
        """Return the quaternion for the given look vector."""
        response.quat = self.kin_solver.find_quaternion_transform(request.point)
        return response


def main(args=None):
    """Run main entry point."""
    rclpy.init(args=args)

    orb_kin_service = OrbitaKinematicsService()

    rclpy.spin(orb_kin_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
