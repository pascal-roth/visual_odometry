import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from vo_pipeline.trajectory import Trajectory


class KeypointTrajectories:
    # TODO: cleanup, see what is not relevant anymore, one file
    def __init__(self, K: np.ndarray, merge_threshold: int = 2):
        self.K = K
        self.landmarks: List[np.ndarray] = None
        self.merge_threshold = merge_threshold
        self.trajectories: Dict[int, Trajectory] = dict()
        self._next_trajectory_idx = 0
        self.on_frame: Dict[int, Dict[int, Trajectory]] = defaultdict(dict)
        self.traj2landmark: Dict[int, int] = dict()
        self.latest_frame = 0

    def set_point_cloud(self, point_cloud: np.ndarray):
        self.landmarks = point_cloud

    def add_pt(self, frame_idx: int, pt: np.ndarray, des: np.ndarray,
               transform: np.ndarray) -> Trajectory:
        # if frame_idx not in self.on_frame:
        trajectory, _ = self._create_trajectory(frame_idx, pt, des, transform)
        return trajectory

        # trajectory, dist = self._find_closest(frame_idx, pt)
        # if dist <= self.merge_threshold:
        #     trajectory.tracked_to(frame_idx, pt, des, transform)
        #     self.on_frame[frame_idx][trajectory.traj_idx] = trajectory
        #     return trajectory
        # else:
        # trajectory, _ = self._create_trajectory(frame_idx, pt, des,
        #                                         transform)
        # return trajectory

    def tracked_to(self,
                   traj_idx: int,
                   frame_idx: int,
                   pt: np.ndarray,
                   des: np.ndarray,
                   transform: np.ndarray,
                   landmark_id=None) -> Trajectory:
        trajectory = self.trajectories[traj_idx]
        trajectory.tracked_to(frame_idx, pt, des, transform)
        self.on_frame[frame_idx][traj_idx] = trajectory
        if landmark_id is not None:
            self.traj2landmark[traj_idx] = landmark_id
        self._next_frame(frame_idx)
        return trajectory

    def latest_keypoints(self) -> Tuple[np.ndarray, List[Trajectory], List[np.ndarray]]:
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

    # def _find_closest(self, frame_idx: int,
    #                   pt: np.ndarray) -> Tuple[Trajectory, float]:
    #     min_dist = np.inf
    #     closest = None
    #     for traj in self.on_frame[frame_idx].values():
    #         dist = np.linalg.norm(traj.final_point - pt)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest = traj
    #     return closest, float(min_dist)

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

    def _next_frame(self, frame_idx: int):
        if frame_idx <= self.latest_frame:
            return
        self.latest_frame = frame_idx
        if self.latest_frame < 6:
            return

        for trajectory in self.on_frame[self.latest_frame - 1].values():
            # only triangulate trajectories lasting longer than 4 frames and it's not been triangulated before
            if trajectory.final_idx - trajectory.init_idx < 6 or trajectory.traj_idx in self.traj2landmark:
                continue
            # TODO: angle dependet when to add new keypoint
            # TODO: change if condition, s.t. if cannot be tracked anymore and if baseline is good enogh add point
            # trajectory could not be tracked anymore
            # ==> triangulate resulting point
            pt = trajectory.triangulate_3d_point()
            self.traj2landmark[trajectory.traj_idx] = len(self.landmarks)
            self.landmarks.append(pt)