from typing import Tuple

import cv2
import numpy as np

from utils.matrix import skew
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from params import *


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
        U, _, VT = np.linalg.svd(E)
        R1 = U @ W @ VT
        R2 = U @ W.T @ VT
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
    def _linear_triangulation(pts1: np.ndarray, pts2: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
        assert pts1.shape == pts2.shape, "The number of matched points in both images has to be the same"
        assert M1.shape == M2.shape == (3, 4), "Homogeneous projection matrices have be of shape (3,4)"
        n_pts, _ = pts1.shape
        P = np.zeros((n_pts, 4))
        for i in range(n_pts):
            A1 = skew(pts1[i, :]) @ M1
            A2 = skew(pts2[i, :]) @ M2
            A = np.vstack((A1, A2))
            _, _, VT = np.linalg.svd(A, full_matrices=False)
            P[i, :] = VT.T[:, -1]

        pts = (P.T / P[:, 3]).T
        return pts

    def _estimate_fundamental(self, normalize=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        matcher = FeatureMatcher(MatcherType.FLANN, k=2, matching_threshold=MATCHING_THRESHOLD)
        matches = matcher.match_descriptors(des0, des1)
        # matcher.match_plotter(self.img1, kp0, self.img2, kp1, matches)

        # 2D hom. matched points (num_matches, 3)
        pts0 = np.array([kp0[match.queryIdx].pt for match in matches])
        pts1 = np.array([kp1[match.trainIdx].pt for match in matches])
        # pts0 = np.loadtxt("./matches0001.txt").T
        # pts1 = np.loadtxt("./matches0002.txt").T
        pts0 = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
        pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))

        def normalize_pts(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            num_pts, _ = pts.shape
            mean = np.mean(pts[:, 0:2], axis=0)
            centered = pts[:, 0:2] - mean
            sigma = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
            s = np.sqrt(2) / sigma
            T = np.array([[s, 0, -s * mean[0]],
                          [0, s, -s * mean[1]],
                          [0, 0, 1]])
            return (T @ pts.T).T, T

        # normalize pts for better numerical conditioning
        if normalize:
            norm_pts0, T_1 = normalize_pts(pts0)
            norm_pts1, T_2 = normalize_pts(pts1)
            F, mask = cv2.findFundamentalMat(norm_pts0[:, 0:2], norm_pts1[:, 0:2], cv2.FM_RANSAC,
                                             ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
                                             confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS)
            # unnormalize fundamental matrix
            F = T_2.T @ F @ T_1
        else:
            F, mask = cv2.findFundamentalMat(pts0[:, 0:2], pts1[:, 0:2], cv2.FM_RANSAC,
                                             ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
                                             confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS)

        # select only inlier points
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
        return F, pts0, pts1

    # def fundamental(self, pts1, pts2):
    #     num_pts, _ = pts1.shape
    #     A = np.zeros((num_pts, 9))
    #     for i in range(num_pts):
    #         p1 = pts1[i, :][np.newaxis].T
    #         p2 = pts2[i, :][np.newaxis].T
    #         k = np.kron(p1, p2)
    #         A[i, :] = k.T
    #     _, _, V = np.linalg.svd(A, full_matrices=False)
    #     F = V[-1, :].reshape(3, 3).T
    #     u, s, v = np.linalg.svd(F)
    #     s[2] = 0
    #     return u @ np.diag(s) @ v
