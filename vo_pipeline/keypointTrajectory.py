import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from params import *
from utils.matrix import hom_inv
from vo_pipeline.poseEstimation import PoseEstimation

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

    def get_active_inactive(self):
        active = []
        inactive = []
        landmark_idx = {
            self.traj2landmark[t]
            for t in self.on_frame[self.latest_frame].keys()
            if t in self.traj2landmark
        }
        for i, landmark in enumerate(self.landmarks):
            if i in landmark_idx:
                active.append(landmark)
            else:
                inactive.append(landmark)
        return active, inactive

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
        baseline_uncertainty = self.baseline_uncertainty(trajectory)
        if baseline_uncertainty is not None and baseline_uncertainty < MIN_BASELINE_UNCERTAINTY:
            return None
        trajectory.tracked_to(frame_idx, pt, des,
                              transform)  # TODO: change that
        self.on_frame[frame_idx][traj_idx] = trajectory
        if landmark_id is not None:
            self.traj2landmark[traj_idx] = landmark_id
        self._next_frame(frame_idx)
        return trajectory

    def baseline_uncertainty(self, trajectory: Trajectory) -> Optional[float]:
        if trajectory.traj_idx not in self.traj2landmark:
            return np.nan
        landmark = self.landmarks[self.traj2landmark[trajectory.traj_idx]]
        landmark_hom = np.hstack((landmark, np.array([1])))[np.newaxis].T
        # T_Ci<-W
        landmark_init = trajectory.init_transform @ landmark_hom
        init = hom_inv(trajectory.init_transform)[0:3, 3]
        final = hom_inv(trajectory.final_transform)[0:3, 3]
        dist = np.linalg.norm(final - init)
        depth = landmark_init[2]
        return float(depth / dist)

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
            if not traj.traj_idx in self.traj2landmark:
                continue
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

    def triangulate_landmarks(self, T: np.ndarray) -> np.ndarray:
        old_landmarks = []
        new_landmarks = []
        old_pts = []
        new_pts = []
        angles = []
        old_trajectories = []
        new_trajectories = []

        for trajectory in self.on_frame[self.latest_frame].values():
            new_landmark = trajectory.traj_idx not in self.traj2landmark
            if not new_landmark:
                old_trajectories.append(trajectory)
                old_landmarks.append(
                    self.landmarks[self.traj2landmark[trajectory.traj_idx]])
                old_pts.append(trajectory.final_point)
                continue

            v1 = np.linalg.inv(
                self.K) @ np.array(list(trajectory.init_point) + [1])
            v2 = np.linalg.inv(
                self.K) @ np.array(list(trajectory.final_point) + [1])
            v1 = np.array(list(v1) + [1])
            v2 = np.array(list(v2) + [1])
            v1 = hom_inv(trajectory.init_transform) @ v1
            v2 = hom_inv(trajectory.final_transform) @ v2
            v1 = v1[0:3]
            v2 = v2[0:3]
            angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.rad2deg(angle)
            large_baseline = angle > MIN_RAY_ANGLE
            # only triangulate trajectories with large baseline and not triangulated before
            if not large_baseline:
                continue
            # ==> triangulate resulting point
            landmark = trajectory.triangulate_3d_point()
            new_trajectories.append(trajectory.traj_idx)
            angles.append(angle)
            new_landmarks.append(landmark)
            new_pts.append(trajectory.final_point)
        if len(new_landmarks) == 0:
            return np.zeros((0))

        # keep only inliers to the last transform
        trajectories = old_trajectories + new_trajectories
        landmarks = old_landmarks + new_landmarks
        pts = old_pts + new_pts
        landmarks = np.hstack((np.asarray(landmarks, dtype=np.float32),
                               np.ones((len(landmarks), 1), dtype=np.float32)))
        pts = np.asarray(pts, dtype=np.float32)
        M = self.K @ T[0:3, 0:4]
        proj = (M @ landmarks.T)
        proj = (proj[0:2, :] / proj[2, :]).T
        err = np.sum((proj - pts)**2, axis=1)
        inliers = err < TRAJECTORY_RANSAC_REPROJ_THESHOLD**2

        for i in range(len(old_landmarks)):
            if not inliers[i]:
                trajectory = trajectories[i]
                del self.traj2landmark[trajectory.traj_idx]

        new_landmarks = []
        for i in range(len(old_landmarks), landmarks.shape[0]):
            if inliers[i]:
                landmark = landmarks[i, 0:3]
                new_landmarks.append(landmark)
                traj_idx = trajectories[i]
                landmark_id = len(self.landmarks)
                self.traj2landmark[traj_idx] = landmark_id
                self.landmarks.append(landmark)
        return np.asarray(new_landmarks)