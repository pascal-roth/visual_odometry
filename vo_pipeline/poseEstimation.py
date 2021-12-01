# function should include:
#   - P3P with RANSAC to get R and T
#   - Heuristic when to acquire new landmarks/ keyframe

# import packages
import cv2 as cv
import numpy as np


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


def PnP():
    """
    OpenCV PnP solver, find homography already gives M back s.t. this function might not be used
    """
    cv.SOLVEPNP_P3P


