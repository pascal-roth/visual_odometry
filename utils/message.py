import time
from typing import List
import numpy as np
import dataclasses

from utils.transform import HomTransform


# @dataclasses.dataclass(init=True, repr=True)
class FeatureMetaData(object):
    """
    Contains necessary information of a feature for easy access.
    """
    __slots__ = ["id", "response", "lifetime", "cam0_point"]
    id: int
    response: float
    lifetime: int
    cam0_point: np.ndarray


@dataclasses.dataclass(init=True, repr=True)
class FeatureMeasurement(object):
    """
    Mono measurement of a feature.
    """
    __slots__ = ["id", "u", "v"]
    id: int
    u: float
    v: float

    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v])


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
    __slots__ = ["timestamp", "pose", "velocity", "cam_pose"]
    timestamp: float
    pose: HomTransform
    velocity: np.ndarray
    cam_pose: HomTransform


@dataclasses.dataclass(init=True, repr=True)
class LandmarkData:
    timestamp: float
    landmarks: np.ndarray
