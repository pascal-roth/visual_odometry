import time
import numpy as np


class IMUMessage:
    timestamp: time.time
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


class FeatureMessage:
    pass


class ImageMessage:
    timestamp: time.time
    image: np.ndarray
