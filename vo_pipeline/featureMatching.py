import enum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Callable


# CV Tutorial
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html


class MatcherType(enum.Enum):
    # brute force matching
    BF = 0,
    # Fast KNN matching
    FLANN = 1,


class FeatureMatcher:
    def __init__(self, matcher_type: MatcherType,
                 k: int = 2,
                 nbr_tress: int = 5,
                 nbr_search: int = 50,
                 matching_threshold: float = 0.8):
        """
        :param matcher_type:        matcher algorithm used
        :param k:                   Number of Nearest Neighbors
        :param nbr_tress:           number of trees
        :param nbr_search:          number of times trees are recursively traversed, higher values = higher precision
        :param matching_threshold   discard all matches where first and second match are too close together
        """
        # save parameters
        self.k = k
        self.nbr_tress = nbr_tress
        self.nbr_search = nbr_search
        self.matching_threshold = matching_threshold

        # get matcher
        self.matcher: Callable
        self.matcher_type = matcher_type
        self.get_matcher()

    def get_matcher(self):
        if self.matcher_type == MatcherType.BF:
            self.matcher = cv.BFMatcher()
        elif self.matcher_type == MatcherType.FLANN:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.nbr_tress)
            search_params = dict(checks=self.nbr_search)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def match_descriptors(self, des1: np.ndarray, des2: np.ndarray, filter: bool = True):
        """
        :param des1:    Descriptors of image 1 / point cloud
        :param des2:    Descriptors of image 2
        :param filter:  if True, only the "good" matches are returned
        """
        matches = self.matcher.knnMatch(des1, des2, k=self.k)
        if filter:
            matches = self.match_filter(matches)

        # only return first column, second closest match only needed for filter operation
        return [match[0] for match in matches]

    def match_filter(self, matches):
        # ratio test from Lowe's paper: Distinctive Image Features from Scale-Invariant Keypoints
        # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        return (match for match in matches if
                match[0].distance < self.matching_threshold * match[1].distance)

    @staticmethod
    def match_plotter(img1, kp1, img2, kp2, matches):
        img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, np.array(matches)[np.newaxis].T, None,
                                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.show()
