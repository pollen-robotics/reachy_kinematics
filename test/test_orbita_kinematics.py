import pytest

import numpy as np
from reachy_kinematics import orbita_kinematics
from geometry_msgs.msg import Quaternion, Point

from reachy_kinematics.orbita import Actuator

class TestOrbitaKinematics():
    @classmethod
    def setup_class(cls):
        cls.solver = orbita_kinematics.OrbitaKinSolver()

    def test_disk_order(self):
        assert self.solver.disks_names == ['disk_top, disk_middle, disk_bottom']

    def test_orbita_ik(self): 
        quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        disks = self.solver.orbita_ik(quat)
        res_disks = list(disks.position)
        assert abs(res_disks[0] - (-0.91)) < 0.01
        assert abs(res_disks[1] - (-1.18)) < 0.01
        assert abs(res_disks[2] - (-1.04)) < 0.01

    def test_look_vector_to_quaternion(self):
        point = Point(x=0.5, y=0.0, z=0.0)
        quat = self.solver.find_quaternion_transform(point)
        assert quat == Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
