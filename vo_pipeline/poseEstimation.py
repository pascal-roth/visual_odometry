# function should include:
#   - P3P with RANSAC to get R and T
#   - Heuristic when to acquire new landmarks/ keyframe

# import packages
import cv2 as cv
import numpy as np
import enum
from typing import Callable, Tuple
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.trackPoints import TrackPoints
import params

class AlgoMethod(enum.Enum):
    DEFAULT = 0,
    P3P = 1,
    AP3P = 2,


class PoseEstimation:
    
    def __init__(self, K: np.ndarray, pointcloud: np.ndarray, first_kps: np.ndarray, use_KLT: bool, algo_method_type: AlgoMethod = AlgoMethod.DEFAULT):
        self.K = K
        self.pointcloud = pointcloud
        self.prev_kpts = first_kps
        self.algo_method_type = algo_method_type
        self.algo_method: Callable
        self.get_method()   
        self.matcher = FeatureMatcher(MatcherType.FLANN, k=2) 
        self.use_KLT = use_KLT
        self.KLT_tracker = TrackPoints(params.KLT_RADIUS, params.KLT_N_ITERS, params.KLT_LAMBDA)
        
    def get_method(self):
        if self.algo_method_type == AlgoMethod.DEFAULT:
            self.algo_method = cv.SOLVEPNP_ITERATIVE()
        elif self.algo_method_type == AlgoMethod.P3P:
            self.algo_method = cv.SOLVEPNP_P3P
        elif self.algo_method_type == AlgoMethod.AP3P:
            self.algo_method = cv.SOLVEPNP_AP3P

    def update_pointcloud_and_kpts(self, pointcloud: np.ndarray, kps: np.ndarray):
        self.pointcloud = pointcloud
        self.prev_kpts = kps
    
    def PnP(self, img_key_points: np.ndarray) -> np.ndarray:
        
        """
        :param img_key_points:      matched keypoints from current image
        :return                     M matrix
        """
        # Same number of keypoints
        assert self.pointcloud.shape[0] == img_key_points.shape[0]
        
        # Solve RANSAC P3P to extract rotation matrix and translation vector 
        success, rvec, trans, _ = cv.solvePnPRansac(self.pointcloud, img_key_points, self.K, distCoeffs=None, flags=self.algo_method)
        assert success
        
        # Convert to homogeneous coordinates
        R, _ = cv.Rodrigues(rvec)
        M = np.eye(4)
        M[0:3, 0.3] = R
        M[0:3, 3] = trans
        return M
        
    def match_key_points(self, pointcloud: np.ndarray, kp0: np.ndarray, des0: np.ndarray, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:

        """
        :param pointcloud:          Initial pointlcoud
        :param kp0:                 keypoints from last keyframe
        :param des0:                descriptors from last keyframe
        :param img0:                last keyframe image
        :param img1:                current image
        :return                     tuple with matched pointcloud and keypoints from current image
        """

        if self.use_KLT:
            # Apply KLT tracking to get keypoints in img1
            
            # Solution from  exercise9
            # pts1, mask = self.KLT_tracker.trackKLT(img0, img1, kp0)
            # matched_pointcloud = pointcloud[mask]
            
            # Opencv KLT
            pts1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, np.round(self.prev_kpts), None, maxLevel=params.KLT_NUM_PYRAMIDS)
            found = st == 1
            pts1 = pts1[found[:, 0]]
            self.update_pointcloud_and_kpts(self.pointcloud[found[:, 0], 0:3], pts1)

        else:
            # Extract keypoints and descriptors from next image
            descriptor = FeatureExtractor(ExtractorType.SIFT)
            kp1, des1 = descriptor.get_kp(img1)
            matches = self.matcher.match_descriptors(des0, des1)

            # 2D pixel coordinates and matched points in pointcloud
            num_matches = len(matches)
            pts1 = np.zeros((num_matches, 2))
            matched_pointcloud = np.zeros((num_matches, 3))

            # Save matched keypoints from current image and corresponding points in pointcloud
            for i, match in enumerate(matches):
                pts1[i, :] = kp1[match.trainIdx].pt
                matched_pointcloud[i, :] = pointcloud[match.queryIdx, :]
            self.update_pointcloud_and_kpts(matched_pointcloud, pts1)

        return pts1

                


