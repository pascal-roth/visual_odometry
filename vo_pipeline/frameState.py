import numpy as np
from typing import Dict


class FrameState:
    # idx: int
    # img: np.ndarray
    # pose: np.ndarray
    # keypoints: np.ndarray  # 2 x N matrix
    # descriptors: np.ndarray  # 16 x N matrix
    # pointcloud: np.ndarray  # 3 x N matrix
    # candidate_C: np.ndarray  # 2 x M matrix
    # candidate_F: np.ndarray  # 2 x M matrix
    # initial_poses: Tuple[np.ndarray]  # 12x M matrix

    def __init__(self,
                 idx: int,
                 img: np.ndarray,
                 pose: np.ndarray,
                 keypoints: np.ndarray = None,
                 descriptors: np.ndarray = None):
        assert idx >= 0, "Frame index must be non-negative"
        self.idx = idx
        self.img = img
        assert pose.shape == (
            4, 4), "pose has to be a homogeneous (4, 4) tranfrom matrix"
        self.pose = pose
        self.keypoints = keypoints
        self.descriptors = descriptors


        self.keypoints: Dict[int, np.ndarray] = dict()