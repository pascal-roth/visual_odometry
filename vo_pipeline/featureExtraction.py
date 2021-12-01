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
            self.extractor = cv.SIFT_create()

    def get_kp(self, img: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        """
        Get keypoints in img according to the set ExtractorType
        :param img:
        :return: list of keypoints, features as (num_keypoints, 128) np.ndarray
        """
        kp, des = self.extractor.detectAndCompute(img, None)
        return kp, des

