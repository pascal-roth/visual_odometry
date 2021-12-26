import time
from typing import List
import numpy as np
import dataclasses

from utils.transform import HomTransform


class FeatureMetaData(object):
    """
    Contains necessary information of a feature for easy access.
    """
    def __init__(self):
        self.id: int = None  # int
        self.response: float = None  # float
        self.lifetime: int = None  # int
        self.cam0_point: np.ndarray = None  # vec2


class FeatureMeasurement(object):
    """
    Mono measurement of a feature.
    """
    def __init__(self):
        self.id = None
        self.u0 = None
        self.v0 = None
        self.u1 = None
        self.v1 = None


@dataclasses.dataclass(init=True, repr=True)
class FeatureData:
    __slots__ = ["timestamp", "features"]
    timestamp: float
    features: List[FeatureMeasurement]


@dataclasses.dataclass(init=True, repr=True)
class FrameData:
    __slots__ = ["timestamp", "image"]
    timestamp: float
    image: np.ndarray


@dataclasses.dataclass(init=True, repr=True)
class IMUData:
    __slots__ = ["timestamp", "angular_velocity", "linear_acceleration"]
    timestamp: float
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


@dataclasses.dataclass(init=True, repr=True)
class PoseData:
    __slots__ = ["timestamp", "pose", "velocity", "cam0_pose"]
    timestamp: float
    pose: HomTransform
    velocity: np.ndarray
    cam0_pose: HomTransform
