# import packages
import numpy as np
import cv2 as cv


class SIFT:
    def __init__(self):
        self.sift = cv.SIFT_create()

    def get_kp(self, img: np.ndarray):
        kp, des = self.sift.detectAndCompute(img, None)
        return kp, des

