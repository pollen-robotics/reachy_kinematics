"""ROS2 Foxy package for Reachy orbita kinematics.

See README.md for details on the services exposed.

"""
import rclpy
from rclpy.node import Node

from reachy_msgs.srv import GetOrbitaIK, GetQuaternionTransform

from .orbita_kinematics import OrbitaKinematicsSolver


class OrbitaKinematicsService(Node):
    """Node exposing the reachy orbita kinematics services.

    Services:
    - /orbita/kinematics/inverse reachy_msgs/srv/GetOrbitaIK.srv
    - /orbita/kinematics/look_vector_to_quaternion reachy_msgs/srv/GetQuaternionTransform.srv
    """

    def __init__(self):
        """Set up the node."""
        super().__init__('orbita_kinematics_service')
        self.logger = self.get_logger()
        self.kin_solver = OrbitaKinematicsSolver()
        self.ik_service = self.create_service(GetOrbitaIK, 'orbita/kinematics/inverse', self.ik_callback)
        self.logger.info(f'Starting service "{self.ik_service.srv_name}".')
        self.orbita_look_at_tf_service = self.create_service(GetQuaternionTransform, 'orbita/kinematics/look_vector_to_quaternion', self.vec2quat_callback)
        self.logger.info(f'Starting service "{self.orbita_look_at_tf_service.srv_name}".')

        self.logger.info('Node ready!')

    def ik_callback(self, request: GetOrbitaIK.Request, response: GetOrbitaIK.Response):
        """Compute the inverse kinematics for orbita given the request."""
        try:
            response.disk_position = self.kin_solver.orbita_ik(request.orientation)
            response.success = True
        except ValueError:
            self.logger.warning(f'Math domain error after ik request: {request.orientation}')
            response.success = False
        return response

    def vec2quat_callback(self, request: GetQuaternionTransform.Request, response: GetQuaternionTransform.Response):
        """Return the quaternion for the given look vector."""
        response.orientation = self.kin_solver.find_quaternion_transform(request.look_vector)
        return response


def main(args=None):
    """Run main entry point."""
    rclpy.init(args=args)

    orb_kin_service = OrbitaKinematicsService()
    rclpy.spin(orb_kin_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
