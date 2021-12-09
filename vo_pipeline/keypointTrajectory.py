import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from vo_pipeline.trajectory import Trajectory


class KeypointTrajectories:
    def __init__(self):
        self.landmarks: List[np.ndarray] = []
        self.trajectories: Dict[int, Trajectory] = dict()
        self._next_trajectory_idx = 0
        self.on_frame: Dict[int, Dict[int, Trajectory]] = defaultdict(dict)
        self.traj2landmark: Dict[int, int] = dict()
        self.latest_frame = 0

    def get_active(self) -> List[np.ndarray]:
        active = []
        active = [
            self.landmarks[self.traj2landmark[t]]
            for t in self.on_frame[self.latest_frame].keys()
            if t in self.traj2landmark
        ]
        return active

    def tracked_to(self,
                   traj_idx: int,
                   frame_idx: int,
                   pt: np.ndarray,
                   transform: np.ndarray,
                   landmark_id=None) -> Trajectory:
        """Tracks the given trajectory to the given frame

        Args:
            traj_idx (int): Trajectory index to track
            frame_idx (int): frame idx to track to
            pt (np.ndarray): 2D image point
            transform (np.ndarray): homogeneous transform from world coordinates to camera coordinate
            landmark_id ([type], optional): index into self.landmarks defining associated 3D landmark. Defaults to None.

        Returns:
            Trajectory: updated tracked trajectory
        """
        trajectory = self.trajectories[traj_idx]
        trajectory.tracked_to(frame_idx, pt, transform)
        self.on_frame[frame_idx][traj_idx] = trajectory
        if landmark_id is not None:
            self.traj2landmark[traj_idx] = landmark_id
        self._next_frame(frame_idx)
        return trajectory

    # def baseline_uncertainty(self, trajectory: Trajectory) -> Optional[float]:
    #     if trajectory.traj_idx not in self.traj2landmark:
    #         return np.nan
    #     landmark = self.landmarks[self.traj2landmark[trajectory.traj_idx]]
    #     landmark_hom = np.hstack((landmark, np.array([1])))[np.newaxis].T
    #     # T_Ci<-W
    #     landmark_init = trajectory.init_transform @ landmark_hom
    #     init = hom_inv(trajectory.init_transform)[0:3, 3]
    #     final = hom_inv(trajectory.final_transform)[0:3, 3]
    #     dist = np.linalg.norm(final - init)
    #     depth = landmark_init[2]
    #     return float(depth / dist)

    def at_frame(
            self,
            frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keypoints with known landmarks in the latest frame
        
        Args:
            frame_idx (int): current frame

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: keypoints (N, 2), trajectory indices (N,), landmarks (N, 3)
        """
        keypoints = []
        trajectories = []
        landmarks = []
        assert frame_idx in self.on_frame, f"frame_idx {frame_idx} needs to have keypoints associated with it"
        for trajectory in self.on_frame[frame_idx].values():
            if trajectory.traj_idx in self.traj2landmark:
                traj_frame_idx = frame_idx - trajectory.init_idx
                assert 0 <= traj_frame_idx <= trajectory.final_idx - trajectory.init_idx
                keypoints.append(trajectory.pts[traj_frame_idx])
                trajectories.append(trajectory.traj_idx)
                landmark = self.landmarks[self.traj2landmark[
                    trajectory.traj_idx]]
                landmarks.append(landmark)

        return np.asarray(keypoints, dtype=np.float32), np.asarray(
            trajectories, dtype=np.int32), np.asarray(landmarks,
                                                      dtype=np.float32)

    def mean_trajectory_length(self) -> float:
        lengths = []
        for traj in self.on_frame[self.latest_frame].values():
            if not traj.traj_idx in self.traj2landmark:
                continue
            lengths.append(traj.final_idx - traj.init_idx)
        return np.mean(lengths)

    def create_trajectory(self, frame_idx: int, pt: np.ndarray,
                          transform: np.ndarray) -> Tuple[Trajectory, int]:
        """Create a new trajectory

        Args:
            frame_idx (int): current frame
            pt (np.ndarray): 2D image coordinates of new point
            transform (np.ndarray): homogeneous transform from world coordinates to camera coordinate

        Returns:
            Tuple[Trajectory, int]: created trajectory and it's id
        """
        traj_idx = self._next_trajectory_idx
        self._next_trajectory_idx += 1
        trajectory = Trajectory(traj_idx, frame_idx, pt, transform)
        self.trajectories[traj_idx] = trajectory
        self.on_frame[frame_idx][traj_idx] = trajectory
        self._next_frame(frame_idx)
        return trajectory, traj_idx

    def _next_frame(self, frame_idx: int):
        """Increments internal frame counter

        Args:
            frame_idx (int): current frame idx
        """
        self.latest_frame = max(self.latest_frame, frame_idx)
