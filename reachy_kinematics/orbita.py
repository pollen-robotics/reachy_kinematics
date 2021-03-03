import numpy as np

from pyquaternion import Quaternion
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R


def rot(axis, deg):
    """Compute 3D rotation matrix given euler rotation."""
    return R.from_euler(axis, np.deg2rad(deg)).as_matrix()


class Actuator(object):
    """
    This actuator is composed of three disks, linked to three arms and a
    platform in the end. The goal is to orientate the platform, so the disks do
    a rotation following a circle called "proximal circle".
    Then, these disks make the arm rotate around the platform's center on a
    circle called "distal circle".
    Three parameters need to be set : The distal radius R and the 3D
    coordinates of the centers of the distal circle and the proximal circle.
    The mathematical explanation can be found in the spherical_symbolic.ipynb
    notebook
    """

    def __init__(self,
                 Pc_z=[0, 0, 89.4],
                 Cp_z=[0, 0, 64.227], R=39.162,
                 R0=np.dot(rot('z', 60), rot('y', 10))):
        self.Pc_z = np.array(Pc_z)
        self.Cp_z = np.array(Cp_z)
        self.R = R
        self.x0, self.y0, self.z0 = np.array(R0)

        self.x0_quat = Quaternion(0, self.x0[0], self.x0[1], self.x0[2])
        self.y0_quat = Quaternion(0, self.y0[0], self.y0[1], self.y0[2])
        self.z0_quat = Quaternion(0, self.z0[0], self.z0[1], self.z0[2])

        self.last_angles = np.array([0, 2 * np.pi / 3, -2 * np.pi / 3])
        self.offset = np.array([0, 0, 0])

    def get_new_frame_from_vector(self, vector, angle=0):
        """
        Compute the coordinates of the vectors of a new frame whose Z axis is
        the chosen vector
        Parameters
        ----------
        vector : array_like
            Vector used to orientate the platform
        angle : float
            The desired angle of rotation of the platform on its Z axis
            in degrees
        Returns
        -------
        X : array_like
            New X vector of the platform's frame
        Y : array_like
            New Y vector of the platform's frame
        Z : array_like
            New Z vector of the platform's frame
        """

        beta = np.deg2rad(angle)

        # GOAL VECTOR (the desired Z axis)
        goal = vector
        goal_norm = [
            i / LA.norm(goal) for i in goal
        ]

        alpha = np.arccos(np.vdot(self.z0, goal_norm))  # Angle of rotation

        if alpha == 0:
            v = Quaternion(0.0, 0.0, 0.0, 1.0)

        else:  # Vector of rotation as a quaternion
            # VECTOR AND ANGLE OF ROTATION
            vec = np.cross(self.z0, goal_norm)
            vector_norm = [
                i / LA.norm(vec) for i in vec
            ]
            v = Quaternion(0.0, vector_norm[0], vector_norm[1], vector_norm[2])

        # QUATERNION OF ROTATION ###
        w1 = np.cos(alpha / 2.0)
        x1 = np.sin(alpha / 2.0) * v.x
        y1 = np.sin(alpha / 2.0) * v.y
        z1 = np.sin(alpha / 2.0) * v.z

        q1 = Quaternion(w1, x1, y1, z1)  # 1st rotation quaternion

        z_prime = q1 * self.z0_quat * q1.inverse

        w2 = np.cos(beta / 2.0)
        x2 = np.sin(beta / 2.0) * z_prime.x
        y2 = np.sin(beta / 2.0) * z_prime.y
        z2 = np.sin(beta / 2.0) * z_prime.z

        # Quaternion of the rotation on new z axis
        q2 = Quaternion(w2, x2, y2, z2)

        new_z = q2 * z_prime * q2.inverse  # Final Z
        new_x = q2 * (q1 * self.x0_quat * q1.inverse) * q2.inverse  # Final X
        new_y = q2 * (q1 * self.y0_quat * q1.inverse) * q2.inverse  # Final Y

        X = np.array([new_x.x, new_x.y, new_x.z])
        Y = np.array([new_y.x, new_y.y, new_y.z])
        Z = np.array([new_z.x, new_z.y, new_z.z])

        return X, Y, Z

    def _eq(self, X, Y, Z):
        R = self.R
        Pc = self.Pc_z
        C = self.Cp_z

        d1 = (
            R**2 * X[2]**2 +
            R**2 * Z[2]**2 -
            C[2]**2 + 2 * C[2] * Pc[2] - Pc[2]**2
        )
        if d1 < 0:
            raise ValueError('math domain error')

        d1 = np.sqrt(d1)

        x11 = R * X[2] - d1
        x12 = R * X[2] + d1
        x2 = R * Z[2] + C[2] - Pc[2]

        sol1 = 2 * np.arctan2(x11, x2)
        sol2 = 2 * np.arctan2(x12, x2)

        if 0 <= np.rad2deg(sol1) <= 180:
            q3 = sol1
        else:
            q3 = sol2

        q1 = np.arctan2(
            Z[1] * np.cos(q3) + X[1] * np.sin(q3),
            Z[0] * np.cos(q3) + X[0] * np.sin(q3),
        )
        return q3, q1

    def get_angles_from_vector(self, vector, angle=0):  # noqa: C901
        """
        Compute the angles of the disks needed to rotate the platform to the
        new frame, using the get_new_frame_from_vector function.
        The expression of q3 and q1 angles are found with the notebook
        spherical_symbolic.ipynb
        Parameters
        ----------
        vector : array_like
            Vector used to orientate the platform
        angle : float
            The desired angle of rotation of the platform on its Z axis
            in degrees
        Returns
        -------
        q11 : float
            angle of the top disk in degrees
        q12 : float
            angle of the middle disk in degrees
        q13 : float
            angle of the bottom disk in degrees
        """

        get_frame = self.get_new_frame_from_vector

        q31_0, q11_0 = self._eq(*get_frame(vector, 0))
        q32_0, q12_0 = self._eq(*get_frame(vector, 120))
        q33_0, q13_0 = self._eq(*get_frame(vector, -120))

        q31, q11 = self._eq(*get_frame(vector, angle))
        q32, q12 = self._eq(*get_frame(vector, angle + 120))
        q33, q13 = self._eq(*get_frame(vector, angle - 120))

        # If there is a discontinuity, add or remove 2*pi radians
        # wrt the sign of angle
        if angle > 0:
            if q11 < q11_0:
                q11 += 2 * np.pi
            if q12 < q12_0:
                q12 += 2 * np.pi
            if q13 < q13_0:
                q13 += 2 * np.pi

        if angle < 0:
            if q11 > q11_0:
                q11 -= 2 * np.pi
            if q12 > q12_0:
                q12 -= 2 * np.pi
            if q13 > q13_0:
                q13 -= 2 * np.pi

        q11 = np.rad2deg(q11)
        q12 = np.rad2deg(q12) - 120
        q13 = np.rad2deg(q13) + 120

        # If the difference between current position and 360° is low,
        # add or remove 360° to the offset applied on disks positions depending
        # on the sign of this difference
        if abs(self.last_angles[0] - q11) >= 180:
            self.offset[0] += np.sign(self.last_angles[0] - q11) * 360

        if abs(self.last_angles[1] - q12) >= 180:
            self.offset[1] += np.sign(self.last_angles[1] - q12) * 360

        if abs(self.last_angles[2] - q13) >= 180:
            self.offset[2] += np.sign(self.last_angles[2] - q13) * 360

        self.last_angles = np.array([q11, q12, q13])

        q11 += self.offset[0]
        q12 += self.offset[1]
        q13 += self.offset[2]

        return q11, q12, q13

    def get_new_frame_from_quaternion(self, qw, qx, qy, qz):
        """
        Compute the coordinates of the vectors of a new frame got by a rotation
        represented by a quaternion
        Parameters
        ----------
        qw : float
            w parameter of the quaternion used to rotate the platform
        qx : float
            x parameter of the quaternion used to rotate the platform
        qy : float
            y parameter of the quaternion used to rotate the platform
        qz : float
            z parameter of the quaternion used to rotate the platform
        Returns
        -------
        X : array_like
            New X vector of the platform's frame
        Y : array_like
            New Y vector of the platform's frame
        Z : array_like
            New Z vector of the platform's frame
        """

        q1 = Quaternion(qw, qx, qy, qz)
        q1_inv = q1.inverse

        new_z = q1 * self.z0_quat * q1_inv  # Final Z
        new_x = q1 * self.x0_quat * q1_inv  # Final X
        new_y = q1 * self.y0_quat * q1_inv  # Final Y

        X = np.array([new_x.x, new_x.y, new_x.z])
        Y = np.array([new_y.x, new_y.y, new_y.z])
        Z = np.array([new_z.x, new_z.y, new_z.z])

        return X, Y, Z

    # FIXME: too complex
    def get_angles_from_quaternion(self, qw, qx, qy, qz):  # noqa: C901
        """
        Compute the angles of the disks needed to rotate the platform to the
        new frame, using the get_new_frame_from_vector function.
        The expression of q3 and q1 angles are found with the notebook
        spherical_symbolic.ipynb
        Parameters
        ----------
        qw : float
            w parameter of the quaternion used to rotate the platform
        qx : float
            x parameter of the quaternion used to rotate the platform
        qy : float
            y parameter of the quaternion used to rotate the platform
        qz : float
            z parameter of the quaternion used to rotate the platform
        Returns
        -------
        q11 : float
            angle of the top disk in degrees
        q12 : float
            angle of the middle disk in degrees
        q13 : float
            angle of the bottom disk in degrees
        """

        def get_frame(q):
            return self.get_new_frame_from_quaternion(q.w, q.x, q.y, q.z)

        quat = Quaternion(qw, qx, qy, qz)
        q31, q11 = self._eq(*get_frame(quat))

        # Find q32 and q12
        # Add an offset of +120°
        w_offset = np.cos(2 * np.pi / 6.0)
        x_offset = np.sin(2 * np.pi / 6.0) * self.z0_quat.x
        y_offset = np.sin(2 * np.pi / 6.0) * self.z0_quat.y
        z_offset = np.sin(2 * np.pi / 6.0) * self.z0_quat.z
        q_offset = Quaternion(w_offset, x_offset, y_offset, z_offset)
        Q = quat * q_offset
        q32, q12 = self._eq(*get_frame(Q))

        # Find q33 and q13
        # Add an offset of -120°
        w_offset = np.cos(-2 * np.pi / 6.0)
        x_offset = np.sin(-2 * np.pi / 6.0) * self.z0_quat.x
        y_offset = np.sin(-2 * np.pi / 6.0) * self.z0_quat.y
        z_offset = np.sin(-2 * np.pi / 6.0) * self.z0_quat.z
        q_offset = Quaternion(w_offset, x_offset, y_offset, z_offset)
        Q = quat * q_offset
        q33, q13 = self._eq(*get_frame(Q))

        last_angles = self.last_angles

        # If there are discontinuities, add or remove 2*pi radians depending on
        # The sign of the last angles
        if (abs(q11 - last_angles[0]) >= 2.96):
            if last_angles[0] > 0:
                q11 += 2 * np.pi
            elif last_angles[0] < 0:
                q11 -= 2 * np.pi
        if (abs(q12 - last_angles[1]) >= 2.96):
            if last_angles[1] > 0:
                q12 += 2 * np.pi
            elif last_angles[1] < 0:
                q12 -= 2 * np.pi
        if (abs(q13 - last_angles[2]) >= 2.96):
            if last_angles[2] > 0:
                q13 += 2 * np.pi
            elif last_angles[2] < 0:
                q13 -= 2 * np.pi

        self.last_angles = np.array([q11, q12, q13])

        return (
            np.rad2deg(q11),
            np.rad2deg(q12) - 120,
            np.rad2deg(q13) + 120,
        )

    def find_quaternion_transform(self, vect_origin, vect_target):
        vo = np.array(vect_origin)
        if np.any(vo):
            vo = vo / LA.norm(vo)

        vt = np.array(vect_target)
        if np.any(vt):
            vt = vt / LA.norm(vt)

        V = np.cross(vo, vt)
        if np.any(V):
            V = V / LA.norm(V)

        alpha = np.arccos(np.dot(vo, vt))
        if np.isnan(alpha) or alpha < 1e-6:
            return Quaternion(1, 0, 0, 0)

        return Quaternion(axis=V, radians=alpha)
