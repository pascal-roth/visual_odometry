import numpy as np
from utils.matrix import skew_3, hom_inv


class Trajectory:
    # TODO: cleanup, see what is not relevant anymore, one file
    def __init__(self, traj_idx: int, init_idx: int, pt: np.ndarray,
                 des: np.ndarray, transform: np.ndarray, K: np.ndarray):
        self.traj_idx = traj_idx
        self.init_idx = init_idx
        self.init_point = pt
        self.init_descriptor = des
        self.init_transform = np.float32(transform)

        self.final_idx: int = init_idx
        self.final_point: np.ndarray = pt
        self.final_descriptor: np.ndarray = des
        self.final_transform: np.ndarray = np.float32(transform)
        self.K = K

    def tracked_to(self, idx: int, pt: np.ndarray, des: np.ndarray,
                   transform: np.ndarray):
        self.final_idx = idx
        self.final_point = pt
        self.final_descriptor = des
        self.final_transform = np.float32(transform)

    def triangulate_3d_point(self) -> np.ndarray:
        M1 = self.K @ self.init_transform[0:3, 0:4]
        M2 = self.K @ self.final_transform[0:3, 0:4]
        A1 = skew_3(self.init_point[0], self.init_point[1], 1) @ M1
        A2 = skew_3(self.final_point[0], self.final_point[1], 1) @ M2
        A = np.vstack((A1, A2))
        _, _, VT = np.linalg.svd(A, full_matrices=False)
        P = VT.T[:, -1]

        # homogenize 3d point
        hom = P[0:3] / P[3]
        # pt1 =  M1 @ P
        # pt1 = pt1 / pt1[2]
        # pt2 = M2 @ P
        # pt2 = pt2 / pt2[2]
        return hom
