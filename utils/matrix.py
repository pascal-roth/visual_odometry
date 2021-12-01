import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    returns a skew symmetric matrix from the vector m
    :param v: 3x1 vector
    :return: 3x3 matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def unskew(m: np.ndarray) -> np.ndarray:
    """
    returns a 3x1 vector from a 3x3 skew symmetric matrix
    :param m: 3x3 skew symmetric matrix
    :return: 3x1 vector
    """
    return np.array([m[2, 1], m[0, 2], m[1, 0]]).reshape((3, 1))
