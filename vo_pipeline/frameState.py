import numpy as np
from typing import Dict


class FrameState:

    def __init__(self,
                 idx: int,
                 img: np.ndarray,
                 pose: np.ndarray,
                 keypoints: np.ndarray = None,
                 descriptors: np.ndarray = None, is_key=False):
        assert idx >= 0, "Frame index must be non-negative"
        self.idx = idx
        self.img = img
        assert pose.shape == (
            4, 4), "pose has to be a homogeneous (4, 4) tranfrom matrix"
        self.pose = pose
        # TODO: resolve conflict which keypoints is the right one
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.keypoints: Dict[int, np.ndarray] = dict()
        self.is_key = is_key