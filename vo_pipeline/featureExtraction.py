import enum
import numpy as np
import cv2 as cv
from typing import Callable, Tuple, List, Any


class ExtractorType(enum.Enum):
    SIFT = 0,


class FeatureExtractor:
    """
    Computes image features for a given ExtractorType
    """
    def __init__(self, extractor_type: ExtractorType = ExtractorType.SIFT):
        self.extractor_type = extractor_type
        self.extractor: Callable
        self.get_extractor()

    def get_extractor(self):
        if self.extractor_type == ExtractorType.SIFT:
            self.extractor = cv.SIFT_create(nfeatures=0,
                                            nOctaveLayers=4,
                                            contrastThreshold=0.03,
                                            edgeThreshold=10,
                                            sigma=1.6)

    def get_kp(self, img: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        """
        Get keypoints in img according to the set ExtractorType
        :param img:
        :return: list of keypoints, features as (num_keypoints, 128) np.ndarray
        """
        kp, des = self.extractor.detectAndCompute(img, None)
        return kp, des

    @staticmethod
    def harris(img: np.ndarray) -> np.ndarray:
        img = np.float32(img)
        dst = cv.cornerHarris(img, 2, 3, 0.1)
        dst = cv.dilate(dst, None)
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50,
                    0.001)
        corners = cv.cornerSubPix(img, np.float32(centroids), (5, 5), (-1, -1),
                                  criteria)
        return corners
