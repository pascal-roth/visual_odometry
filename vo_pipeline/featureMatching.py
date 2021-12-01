# import packages
import enum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Callable


class MatcherType(enum.Enum):
    BF = 0,
    FLANN = 1,


class FeatureMatcher:
    def __init__(self, matcher_type: MatcherType,
                 k: int = 2,
                 nbr_tress: int = 5,
                 nbr_search: int = 50):
        """
        :param matcher_type:    matcher algorithm used
        :param k:               Number of Nearest Neighbors
        :param nbr_tress:       number of trees
        :param nbr_search:      number of times the trees are recursively traversed, i.e. higher values = higher precision
        """
        # save parameters
        self.k = k
        self.nbr_tress = nbr_tress
        self.nbr_search = nbr_search

        # get machter
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

    def match_des(self, des1: np.ndarray, des2: np.ndarray, filter: bool = True):
        """
        :param filter:  if True, only the "good" matches are returned
        :param des1:    Descriptors of image 1 / point cloud
        :param des2:    Descriptors of image 2
        """
        matches = self.matcher.knnMatch(des1, des2, k=self.k)
        if filter:
            matches = self.match_filter(matches)

        return matches

    def match_filter(self, matches) -> np.ndarray:
        matchesMask = np.zeros(len(matches))
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = 1
        matches = np.array(matches)
        return matches[matchesMask == 1]

    @staticmethod
    def match_plotter(img1, kp1, img2, kp2, matches):
        img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.show()
