import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    returns a skew symmetric matrix from the vector m
    :param v: 3x1 vector
    :return: 3x3 matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float32)

def skew_3(a:float, b:float, c:float) -> np.ndarray:
    """
    returns a skew symmetric matrix from the vector m
    :param v: 3x1 vector
    :return: 3x3 matrix
    """
    return np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]], dtype=np.float32)

def unskew(m: np.ndarray) -> np.ndarray:
    """
    returns a 3x1 vector from a 3x3 skew symmetric matrix
    :param m: 3x3 skew symmetric matrix
    :return: 3x1 vector
    """
    return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=np.float32).reshape((3, 1))

    
def hom_inv(T:np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), "T has to be a homogeneous tranform matrix "
    I = np.zeros_like(T)
    I[0:3, 0:3] = T[0:3, 0:3].T
    I[0:3, 3] = -T[0:3, 0:3].T @ T[0:3, 3]
    return I

def to_hom(pts: np.ndarray) -> np.ndarray:
    n_pts, dim = pts.shape
    dtype = pts.dtype
    ones = np.ones((n_pts, 1), dtype=dtype)
    return np.hstack((pts, ones))

# def to_hom_transform(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
