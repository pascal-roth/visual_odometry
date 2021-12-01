import cv2
import numpy as np
import matplotlib.pyplot as plt
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from utils.matrix import unskew, skew
from typing import Tuple


class BootstrapInitializer:

    def __init__(self, img1: np.ndarray, img2: np.ndarray, K: np.ndarray):
        self.img1 = img1
        self.img2 = img2
        self.K = K

        F, pts1, pts2 = self.estimate_F()
        self.T = self.get_T(F, pts1, pts2)

    def get_T(self, F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Computes homogeneous transform T.
        :param F: Fundamental matrix, (3, 3)
        :param K: Intrinsic camera matrix, (3, 3)
        :return: homogeneous transform T, (4, 4)
        """
        # get essential matrix
        E = self.K.T @ F @ self.K

        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        U, _, V = np.linalg.svd(E)
        R1 = U @ W @ V
        R2 = U @ W.T @ V
        u = U[:, 2]
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
        M1 = self.K @ np.eye(3, 4)
        most_pts_in_front = -np.inf
        for R_hat in [R1, R2]:
            for u_hat in [u, -u]:
                T_hat = np.hstack((R_hat, u_hat[np.newaxis].T))
                M2 = self.K @ T_hat
                P_cam1 = self.linear_triangulation(pts1, pts2, M1, M2)
                P_cam2 = (T_hat @ P_cam1.T).T
                num_in_font = np.sum(P_cam1[:, 2] > 0) + np.sum(P_cam2[:, 2] > 0)
                if num_in_font > most_pts_in_front:
                    R = R_hat
                    t = u_hat
                    most_pts_in_front = num_in_font

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return T

    def linear_triangulation(self, pts1: np.ndarray, pts2: np.ndarray, M1: np.ndarray, M2: np.ndarray,
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
            P[i, :] = V[:, 3]

        pts = (P.T / P[:, 3]).T
        return pts if homoeneous else pts[:, 0:3]

    def estimate_F(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        #
        # def drawlines(img1, img2, lines, pts1, pts2):
        #     ''' img1 - image on which we draw the epilines for the points in img2
        #         lines - corresponding epilines '''
        #     r, c = img1.shape
        #     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        #     for r, pt1, pt2 in zip(lines, pts1, pts2):
        #         color = tuple(np.random.randint(0, 255, 3).tolist())
        #         x0, y0 = map(int, [0, -r[2] / r[1]])
        #         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        #         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        #         img1 = cv2.circle(img1, tuple(np.int32(pt1)), 5, color, -1)
        #         img2 = cv2.circle(img2, tuple(np.int32(pt2)), 5, color, -1)
        #     return img1, img2
        #
        # # Find epilines corresponding to points in right image (second image) and
        # # drawing its lines on left image
        # lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
        # lines1 = lines1.reshape(-1, 3)
        # img5, img6 = drawlines(img0, img1, lines1, pts0, pts1)
        # # Find epilines corresponding to points in left image (first image) and
        # # drawing its lines on right image
        # lines2 = cv2.computeCorrespondEpilines(pts0.reshape(-1, 1, 2), 1, F)
        # lines2 = lines2.reshape(-1, 3)
        # img3, img4 = drawlines(img0, img1, lines2, pts0, pts1)
        # plt.subplot(121), plt.imshow(img5)
        # plt.subplot(122), plt.imshow(img3)
        # plt.show()
