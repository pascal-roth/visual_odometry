# function should include:
#   - P3P with RANSAC to get R and T
#   - Heuristic when to acquire new landmarks/ keyframe

# import packages
import cv2 as cv
import numpy as np
import enum
from typing import Callable, Tuple, Optional
import params


class AlgoMethod(enum.Enum):
    DEFAULT = 0,
    P3P = 1,
    AP3P = 2,


class PoseEstimation:

    @staticmethod
    def KLT(
        img0: np.ndarray,
        img1: np.ndarray,
        prev_kpts: np.ndarray,
        curr_kpts: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply KLT tracking to get keypoints in img1
        """
        tracked_pts, status, err = cv.calcOpticalFlowPyrLK(
            img0,
            img1,
            np.float32(prev_kpts),
            curr_kpts,
            winSize=(params.KLT_RADIUS, params.KLT_RADIUS),
            maxLevel=params.KLT_NUM_PYRAMIDS,
            minEigThreshold=params.KLT_MIN_EIGEN_THRESHOLD,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT,
                      params.KLT_N_ITERS, params.KLT_TRACK_PRECISION),
            flags=cv.OPTFLOW_USE_INITIAL_FLOW)
        return tracked_pts, status
