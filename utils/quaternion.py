# Largely adapted from https://github.com/uoip/stereo_msckf

import numpy as np
from utils.matrix import skew


class Quaternion:
    def __init__(self, q: np.ndarray) -> None:
        assert q.shape == (4, )
        self.q = q

    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(np.array([0, 0, 0, 1]))

    @classmethod
    def from_rotation(cls, R: np.ndarray) -> 'Quaternion':
        """
        Convert a rotation matrix to a quaternion.
        Pay attention to the convention used. The function follows the
        conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
        A Tutorial for Quaternion Algebra", Equation (78).
        The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
        """
        if R[2, 2] < 0:
            if R[0, 0] > R[1, 1]:
                t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
                q = [
                    t, R[0, 1] + R[1, 0], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]
                ]
            else:
                t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
                q = [
                    R[0, 1] + R[1, 0], t, R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]
                ]
        else:
            if R[0, 0] < -R[1, 1]:
                t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
                q = [
                    R[0, 2] + R[2, 0], R[2, 1] + R[1, 2], t, R[0, 1] - R[1, 0]
                ]
            else:
                t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
                q = [
                    R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], t
                ]

        q = np.array(q)  # * 0.5 / np.sqrt(t)
        q /= np.linalg.norm(q)
        return cls(q)

    @classmethod
    def small_angle_quaternion(cls, dtheta: np.ndarray) -> 'Quaternion':
        """
        Convert the vector part of a quaternion to a full quaternion.
        This function is useful to convert delta quaternion which is  
        usually a 3x1 vector to a full quaternion.
        For more details, check Equation (238) and (239) in "Indirect Kalman 
        Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
        """
        dq = dtheta / 2.
        dq_square_norm = dq @ dq

        if dq_square_norm <= 1:
            q = np.array([*dq, np.sqrt(1 - dq_square_norm)])
        else:
            q = np.array([*dq, 1.])
            q /= np.sqrt(1 + dq_square_norm)
        return cls(q)
    
    @classmethod
    def from_two_vectors(cls, v0, v1) -> 'Quaternion':
        """
        Rotation quaternion from v0 to v1.
        """
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        d = v0 @ v1

        # if dot == -1, vectors are nearly opposite
        if d < -0.999999:
            axis = np.cross([1, 0, 0], v0)
            if np.linalg.norm(axis) < 0.000001:
                axis = np.cross([0, 1, 0], v0)
            q = np.array([*axis, 0.])
        elif d > 0.999999:
            q = np.array([0., 0., 0., 1.])
        else:
            s = np.sqrt((1 + d) * 2)
            axis = np.cross(v0, v1)
            vec = axis / s
            w = 0.5 * s
            q = np.array([*vec, w])

        q = cls(q / np.linalg.norm(q))
        return q.conjugate()  # hamilton -> JPL

    def to_rotation(self) -> np.ndarray:
        """
        Convert a quaternion to the corresponding rotation matrix.
        Pay attention to the convention used. The function follows the
        conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
        A Tutorial for Quaternion Algebra", Equation (78).
        The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
        """
        q = self.q / np.linalg.norm(self.q)
        vec = q[:3]
        w = q[3]

        R = (2 * w * w -
             1) * np.identity(3) - 2 * w * skew(vec) + 2 * vec[:, None] * vec
        return R

    def normalized(self) -> 'Quaternion':
        """
        Normalize the given quaternion to unit quaternion.
        """
        return Quaternion(self.q / np.linalg.norm(self.q))

    def conjugate(self) -> 'Quaternion':
        """
        Conjugate of a quaternion.
        """
        return Quaternion(np.array([*-self.q[:3], self.q[3]]))

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Perform q1 * q2
        """
        q1 = self.normalized().q
        q2 = other.normalized().q

        L = np.array([[q1[3], q1[2], -q1[1], q1[0]],
                      [-q1[2], q1[3], q1[0], q1[1]],
                      [q1[1], -q1[0], q1[3], q1[2]],
                      [-q1[0], -q1[1], -q1[2], q1[3]]])

        q = L @ q2
        return Quaternion(q / np.linalg.norm(q))
