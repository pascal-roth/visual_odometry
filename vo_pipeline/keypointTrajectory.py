import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from utils.matrix import hom_inv

from vo_pipeline.trajectory import Trajectory


class KeypointTrajectories:
    def __init__(self, K: np.ndarray, merge_threshold: int = 2):
        self.K = K
        self.landmarks: List[np.ndarray] = None
        self.merge_threshold = merge_threshold
        self.trajectories: Dict[int, Trajectory] = dict()
        self._next_trajectory_idx = 0
        self.on_frame: Dict[int, Dict[int, Trajectory]] = defaultdict(dict)
        self.traj2landmark: Dict[int, int] = dict()
        self.latest_frame = 0

    def set_point_cloud(self, point_cloud: List[np.ndarray]):
        self.landmarks = point_cloud

    def add_pt(self, frame_idx: int, pt: np.ndarray, des: np.ndarray,
               transform: np.ndarray) -> Trajectory:
        trajectory, _ = self._create_trajectory(frame_idx, pt, des, transform)
        return trajectory

    def tracked_to(self,
                   traj_idx: int,
                   frame_idx: int,
                   pt: np.ndarray,
                   des: np.ndarray,
                   transform: np.ndarray,
                   landmark_id=None) -> Trajectory:
        """
        function to call if trajectory can be tracked until the current frame
        """
        trajectory = self.trajectories[traj_idx]
        # trajectory_length = trajectory.final_idx - trajectory.init_idx
        # if trajectory_length > 25:
        #     return None
        trajectory.tracked_to(frame_idx, pt, des, transform)  # TODO: change that
        self.on_frame[frame_idx][traj_idx] = trajectory
        if landmark_id is not None:
            self.traj2landmark[traj_idx] = landmark_id
        self._next_frame(frame_idx)
        return trajectory

    def latest_keypoints(
            self) -> Tuple[np.ndarray, List[Trajectory], List[np.ndarray]]:
        keypoints = []
        trajectories = []
        landmarks = []
        for trajectory in self.on_frame[self.latest_frame].values():
            keypoints.append(trajectory.final_point)
            trajectories.append(trajectory)
            landmark = None
            if trajectory.traj_idx in self.traj2landmark:
                landmark = self.landmarks[self.traj2landmark[
                    trajectory.traj_idx]]
            landmarks.append(landmark)

        return np.asarray(keypoints, dtype=np.float32), trajectories, landmarks

    def mean_trajectory_length(self) -> float:
        lengths = []
        for traj in self.on_frame[self.latest_frame].values():
            lengths.append(traj.final_idx - traj.init_idx)
        return np.mean(lengths)

    def _create_trajectory(self, frame_idx: int, pt: np.ndarray,
                           des: np.ndarray,
                           transform: np.ndarray) -> Tuple[Trajectory, int]:
        traj_idx = self._next_trajectory_idx
        self._next_trajectory_idx += 1
        trajectory = Trajectory(traj_idx, frame_idx, pt, des, transform,
                                self.K)
        self.trajectories[traj_idx] = trajectory
        self.on_frame[frame_idx][traj_idx] = trajectory
        self._next_frame(frame_idx)
        return trajectory, traj_idx
    

    #  investigated that are not tracked anymore
    def _next_frame(self, frame_idx: int):
        if frame_idx <= self.latest_frame:
            return
        self.latest_frame = frame_idx
        if self.latest_frame < 6:
            return

        for trajectory in self.on_frame[self.latest_frame - 1].values():
            v1 = np.linalg.inv(self.K) @ np.array(list(trajectory.init_point) + [1])
            v2 = np.linalg.inv(self.K) @ np.array(list(trajectory.final_point) + [1])
            v1 = np.array(list(v1) + [1])
            v2 = np.array(list(v2) + [1])
            v1 = hom_inv(trajectory.init_transform) @ v1
            v2 = hom_inv(trajectory.final_transform) @ v2
            v1 = v1[0:3]
            v2 = v2[0:3]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.rad2deg(angle)
            min_angle = 5
            large_baseline = angle > min_angle
            new_landmark = trajectory.traj_idx not in self.traj2landmark
            # only triangulate trajectories with large baseline and not triangulated before
            if not large_baseline or not new_landmark:
                continue
            # ==> triangulate resulting point
            pt = trajectory.triangulate_3d_point()
            landmark_id = len(self.landmarks)
            self.traj2landmark[trajectory.traj_idx] = landmark_id
            self.landmarks.append(pt)
