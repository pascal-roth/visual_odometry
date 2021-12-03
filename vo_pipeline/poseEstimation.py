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
    
    def __init__(self, K: np.ndarray, use_KLT: bool, algo_method_type: AlgoMethod = AlgoMethod.DEFAULT):
        self.K = K
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
    
    
    def PnP(self, pointcloud: np.ndarray, img_key_points: np.ndarray) -> np.ndarray:
        
        """
        :param pointcloud:          already matched 3D pointcloud extracted from keyframes
        :param img_key_points:      matched keypoints from current image
        :return                     M matrix
        """
        # Same number of keypoints
        assert pointcloud.shape[0] == img_key_points.shape[0]
        
        # Solve RANSAC P3P to extract rotation matrix and translation vector 
        success, rvec, trans, _ = cv.solvePnPRansac(pointcloud, img_key_points, self.K, distCoeffs=None, flags=self.algo_method)
        assert success
        
        # Convert to homogeneous coordinates
        R, _ = cv.Rodrigues(rvec)
        M = np.hstack((R, trans))
        M = np.vstack((M, np.array([0, 0, 0, 1])))

        return M
        
    def match_key_points(self, pointcloud: np.ndarray, kp0: np.ndarray, des0: np.ndarray, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        :param pointcloud:          current pointcloud extracted from last 2 keyframes
        :param kp0:                 keypoints from last keyframe
        :param des0:                descriptors from last keyframe
        :param img0:                last keyframe image
        :param img1:                current image
        :return                     tuple with matched pointcloud and keypoints from current image
        """

        if self.use_KLT:
            # Apply KLT tracking to get keypoints in img1
            pts1, mask = self.KLT_tracker.trackKLT(img0, img1, kp0)
            matched_pointcloud = pointcloud[mask]
            
        
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
                matched_pointcloud[i, :] = pointcloud[match.queryIdx, 0:3]

        return matched_pointcloud, pts1
                


