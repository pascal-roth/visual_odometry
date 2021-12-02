# function should include:
#   - P3P with RANSAC to get R and T
#   - Heuristic when to acquire new landmarks/ keyframe

# import packages
import cv2 as cv
import numpy as np
import enum
from typing import Callable, Tuple
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType


# TODO: finish function
def estimatePose(kp1: np.ndarray, kp2: np.ndarray, matches: np.ndarray):
    """
    Open-CV Tutorial: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

    RANSAC Alternative: According to https://opencv.org/evaluating-opencvs-new-ransacs/, the standard RANSAC method
    has a bad performance, thus USAC_MAGSAC is used! (Paper found [here](https://arxiv.org/abs/1912.05909))

    :param kp1
    :param kp2
    :param matches
    """
    MIN_MATCH_COUNT = 10
    use_USAC = False

    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if use_USAC:
            raise KeyError('Find Homography does not support USAC MAGSAC, have to think about if we implement it'
                           ' by ourself')
            M, mask = cv.findHomography(src_pts, dst_pts, cv.USAC_MAGSAC, 5.0)
        else:
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # h, w, d = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv.perspectiveTransform(pts, M)
        # img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print(f"Not enough matches are found - {len(matches)}/{MIN_MATCH_COUNT}")
        matchesMask = None

    return matchesMask


class AlgoMethod(enum.Enum):
    DEFAULT = 0,
    P3P = 1,
    AP3P = 2,


class PoseEstimation:
    
    def __init__(self, K: np.ndarray, algo_method_type: AlgoMethod=AlgoMethod.DEFAULT):
        self.K = K
        self.algo_method_type = algo_method_type
        self.algo_method: Callable
        self.get_method()   
        self.matcher = FeatureMatcher(MatcherType.FLANN, k=2) 
        
    def get_method(self):
        if self.extractor_type == AlgoMethod.DEFAULT:
            self.algo_method = cv.SOLVEPNP_ITERATIVE()
        elif self.extractor_type == AlgoMethod.P3P:
            self.algo_method = cv.SOLVEPNP_P3P
        elif self.extractor_type == AlgoMethod.AP3P:
            self.algo_method = cv.SOLVEPNP_AP3P
    
    
    def PnP(self, pointcloud: np.ndarray, imgKeypoints: np.ndarray) -> np.ndarray:
        
        """
        :param pointcloud:          already matched 3D pointcloud extracted from keyframes
        :param imgKeypoints:        matched keypoints from current image
        :return                     M matrix
        """
        # Same number of keypoints
        assert pointcloud.shape(1) == imgKeypoints.shape(1)
        
        rot, trans = cv.solvePnPRansac(pointcloud, imgKeypoints, self.K, flags=self.algo_method)
        M = np.column_stack(rot,trans)

        return M
        
    def matchKeyPoints(self, kp0: np.ndarray, kp1: np.ndarray, des0: np.ndarray, des1: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        
        matches = self.matcher.match_descriptors(des0, des1)

        # 2D hom. matched points (num_matches, 3)
        pts0 = np.array([kp0[match.queryIdx].pt for match in matches])
        pts1 = np.array([kp1[match.trainIdx].pt for match in matches])
        pts0 = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
        pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
                


