"""Kinematics solver for Orbita Actuator."""

import numpy as np

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion, Point

from .orbita import Actuator


d_names = ['disk_top, disk_middle, disk_bottom']


class OrbitaKinematicsSolver(object):
    """Kinematics solver class."""

    def __init__(self) -> None:
        """Set up the class."""
        self.disks_names = ['disk_top', 'disk_middle', 'disk_bottom']
        self.orbita = Actuator()

    def orbita_ik(self, quat: Quaternion) -> JointState:
        """Compute inverse kinematics for Orbita.

        Return the disks goal positions to reach the given quaternion.
        """
        disks = JointState()
        disks.name = self.disks_names
        thetas = self.orbita.get_angles_from_quaternion(
            quat.w,
            quat.x,
            quat.y,
            quat.z
        )
        disks.position = list(np.deg2rad(thetas))
        return disks

    def find_quaternion_transform(self, point: Point) -> Quaternion:
        """Transform given goal point into a quaternion."""
        quat = self.orbita.find_quaternion_transform([1, 0, 0], [point.x, point.y, point.z])
        return Quaternion(x=quat.x, y=quat.y, z=quat.z, w=quat.w)
