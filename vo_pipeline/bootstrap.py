from typing import Tuple

import cv2
import numpy as np

from utils.matrix import skew
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType


class BootstrapInitializer:

    def __init__(self, img1: np.ndarray, img2: np.ndarray, K: np.ndarray):
        self.img1 = img1
        self.img2 = img2
        self.K = K

        self.F, self.pts1, self.pts2 = self._estimate_fundamental()
        self.T, self.point_cloud = self._transform_matrix(self.F, self.pts1, self.pts2)

    # def select_baseline(self):
    #     img0 = None
    #     transforms = []
    #     for i, frame in enumerate(self.dataset.frames):
    #         K, img = frame
    #         if i == 0:
    #             img0 = img
    #         else:
    #             F, pts1, pts2 = BootstrapInitializer.estimate_F(img0, img)
    #             T, point_cloud = BootstrapInitializer.get_T(K, F, pts1, pts2)
    #             transforms.append(T)
    #             # if len(transforms) > 1:
    #             # T[0:3, 3] *= len(transforms)
    #             uncertainty = BootstrapInitializer.get_baseline_uncertainty(T, point_cloud)
    #             print(uncertainty)


    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray, point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)

    def _transform_matrix(self, F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes homogeneous transform T.
        :param F: Fundamental matrix, (3, 3)
        :param pts1:
        :param pts2:
        :return: homogeneous transform T, (4, 4), reconstructed point cloud
        """
        # get essential matrix
        E = self.K.T @ F @ self.K

        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        U, _, V = np.linalg.svd(E)
        R1 = U @ W @ V.T
        R2 = U @ W.T @ V.T
        u = U[:, -1]
        if np.linalg.det(R1) < 0:
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2
        if not np.all(u == 0):
            u /= np.linalg.norm(u)

        # disambiguate between the different rotations by selecting the solution where all points are in
        # front of the cameras
        R = None
        t = None
        point_cloud = None

        M1 = self.K @ np.eye(3, 4)
        most_pts_in_front = -np.inf
        for R_hat in [R1, R2]:
            for u_hat in [u, -u]:
                T_hat = np.hstack((R_hat, u_hat[np.newaxis].T))
                M2 = self.K @ T_hat
                P_cam1 = BootstrapInitializer._linear_triangulation(pts1, pts2, M1, M2)
                P_cam2 = (T_hat @ P_cam1.T).T
                num_in_font = np.sum(P_cam1[:, 2] > 0) + np.sum(P_cam2[:, 2] > 0)
                if num_in_font > most_pts_in_front:
                    R = R_hat
                    t = u_hat
                    point_cloud = P_cam1
                    most_pts_in_front = num_in_font

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return T, point_cloud

    @staticmethod
    def _linear_triangulation(pts1: np.ndarray, pts2: np.ndarray, M1: np.ndarray, M2: np.ndarray,
                              homoeneous=True) -> np.ndarray:
        assert pts1.shape == pts2.shape, "The number of matched points in both images has to be the same"
        assert M1.shape == M2.shape == (3, 4), "Homogeneous projection matrices have be of shape (3,4)"
        n_pts, _ = pts1.shape
        P = np.zeros((n_pts, 4))
        for i in range(n_pts):
            A1 = skew(np.array([pts1[i, 0], pts1[i, 1], 1])) @ M1
            A2 = skew(np.array([pts2[i, 0], pts2[i, 1], 1])) @ M2
            A = np.vstack((A1, A2))
            _, _, V = np.linalg.svd(A, full_matrices=False)
            P[i, :] = V[:, -1]

        pts = (P.T / P[:, 3]).T
        return pts if homoeneous else pts[:, 0:3]

    def _estimate_fundamental(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the fundamental matrix F given two images
        (Solves the bootstrapping problem)
        :return: fundamental matrix F, feature points in img1, feature points in img2
        """
        # extract features in both images
        descriptor = FeatureExtractor(ExtractorType.SIFT)
        kp0, des0 = descriptor.get_kp(self.img1)
        kp1, des1 = descriptor.get_kp(self.img2)

        # match features
        matcher = FeatureMatcher(MatcherType.FLANN, k=2)
        matches = matcher.match_descriptors(des0, des1)

        # 2D matched points (num_matches, 2)
        pts0 = np.array([kp0[match.queryIdx].pt for match in matches])
        pts1 = np.array([kp1[match.trainIdx].pt for match in matches])

        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_LMEDS)

        # We select only inlier points
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
        return F, pts0, pts1
