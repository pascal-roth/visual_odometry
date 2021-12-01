# import packages
import enum
import numpy as np
import cv2 as cv
from typing import Callable


class ExtractorType(enum.Enum):
    SIFT = 0,


class FeatureExtractor:
    def __init__(self, extractor_type: ExtractorType = ExtractorType.SIFT):
        self.extractor_type = extractor_type
        self.extractor: Callable
        self.get_extractor()

    def get_extractor(self):
        if self.extractor_type == ExtractorType.SIFT:
            self.extractor = cv.SIFT_create()

    def get_kp(self, img: np.ndarray):
        kp, des = self.extractor.detectAndCompute(img, None)
        return kp, des

