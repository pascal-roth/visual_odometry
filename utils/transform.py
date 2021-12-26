import numpy as np


class HomTransform:
    """
    Represents a Homogeneous Transformation in 3D space.
    """
    def __init__(self, R: np.ndarray, t: np.ndarray):
        assert R is not None
        assert t is not None
        self.R = R
        self.t = t

    @classmethod
    def identity(cls) -> 'HomTransform':
        return cls(np.eye(3), np.zeros(3))

    def to_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[0:3, 0:3] = self.R
        T[0:3, 3] = self.t
        return T

    def inverse(self) -> 'HomTransform':
        return HomTransform(self.R.T, -self.R.T @ self.t)

    def __mul__(self, other: 'HomTransform') -> 'HomTransform':
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return HomTransform(R, t)