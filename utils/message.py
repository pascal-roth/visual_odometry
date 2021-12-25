import time
import numpy as np
import dataclasses

class FeatureMessage:
    pass

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

