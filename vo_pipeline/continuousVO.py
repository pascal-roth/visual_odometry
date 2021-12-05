from typing import Tuple, List, Dict
from utils.loadData import DatasetLoader, DatasetType
from utils.matrix import skew, skew_3
from vo_pipeline import bootstrap
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from vo_pipeline.bootstrap import BootstrapInitializer
from utils.loadData import Dataset
from queue import Queue
from vo_pipeline.frameState import FrameState
from params import *
from collections import defaultdict
import cv2 as cv

import numpy as np


class Trajectory:
    def __init__(self, traj_idx: int, init_idx: int, pt: np.ndarray,
                 des: np.ndarray, transform: np.ndarray, K: np.ndarray):
        self.traj_idx = traj_idx
        self.init_idx = init_idx
        self.init_point = pt
        self.init_descriptor = des
        self.init_transform = np.float32(transform)

        self.final_idx: int = init_idx
        self.final_point: np.ndarray = pt
        self.final_descriptor: np.ndarray = des
        self.final_transform: np.ndarray = np.float32(transform)
        self.K = K

    def tracked_to(self, idx: int, pt: np.ndarray, des: np.ndarray,
                   transform: np.ndarray):
        self.final_idx = idx
        self.final_point = pt
        self.final_descriptor = des
        self.final_transform = np.float32(transform)

    def triangulate_3d_point(self) -> np.ndarray:
        M1 = self.K @ self.init_transform[0:3, 0:4]
        M2 = self.K @ self.final_transform[0:3, 0:4]
        A1 = skew_3(self.init_point[0], self.init_point[1], 1) @ M1
        A2 = skew_3(self.final_point[0], self.final_point[1], 1) @ M2
        A = np.vstack((A1, A2))
        _, _, VT = np.linalg.svd(A, full_matrices=False)
        P = VT.T[:, -1]
        # homogenize 3d point
        return P[0:3] / P[3]


class KeypointTrajectories:
    def __init__(self, K: np.ndarray, merge_threshold=2):
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
            # trajectory could not be tracked anymore
            # ==> triangulate resulting point
            pt = trajectory.triangulate_3d_point()
            self.traj2landmark[trajectory.traj_idx] = len(self.landmarks)
            self.landmarks.append(pt)


class ContinuousVO:
    def __init__(self,
                 dataset: Dataset,
                 descriptorType=ExtractorType.SIFT,
                 matcherType=MatcherType.BF,
                 useKLT=True,
                 algo_method=AlgoMethod.P3P,
                 max_point_distance: int = 50,
                 frames_to_skip: int = 4) -> None:

        self.dataset = dataset
        self.descriptor = FeatureExtractor(descriptorType)
        self.matcher = FeatureMatcher(matcherType)
        self.useKLT = useKLT
        self.poseEstimator = PoseEstimation(self.dataset.K,
                                            use_KLT=useKLT,
                                            algo_method_type=algo_method)

        # in-memory frame buffer
        self.frame_queue: List[FrameState] = []
        self.point_cloud: List[np.ndarray] = None
        # camera calibration matrix
        self.K: np.ndarray = None
        # max point distance for bootstrapp algo
        self.max_point_distance: int = max_point_distance
        # number of frames to skip to get the first baseline
        self.frames_to_skip: int = frames_to_skip
        # save homogeneous coordinates of the last keyframe
        self.last_keyframe_coords: np.ndarray

        self.keypoint_trajectories = KeypointTrajectories(self.dataset.K)
        self.frame_idx = 0

    def step(self):
        K, img = next(self.dataset.frames)
        if self.frame_idx < self.frames_to_skip:
            self.K = K
            self.frame_queue.append(FrameState(self.frame_idx, img, np.eye(4)))
        else:
            self._processFrame(self.frame_idx, img)

        self.frame_idx += 1  

    def _processFrame(self, idx: int, img: np.ndarray):

        keypoints, descriptors = self.descriptor.get_kp(img)

        # bootstrap initialization
        if self.keypoint_trajectories.landmarks is None:
            img1 = self.frame_queue[0].img
            img2 = img
            bootstrapper = BootstrapInitializer(
                img1, img2, self.K, max_point_dist=self.max_point_distance)

            self.keypoint_trajectories.set_point_cloud(
                list(bootstrapper.point_cloud[:, 0:3]))
            # self.poseEstimator.update_pointcloud_and_prev_kpts(
            #     self.point_cloud, keypoints)
            T = bootstrapper.T

            # save img to frame queue
            self.frame_queue.append(FrameState(idx, img, T))

            # initialize keypoint trajectories
            for i in range(bootstrapper.pts1.shape[0]):
                trajectory = self.keypoint_trajectories.add_pt(
                    0, bootstrapper.pts1[i, 0:2], bootstrapper.pts_des1[i],
                    np.eye(4))
                self.keypoint_trajectories.tracked_to(
                    traj_idx=trajectory.traj_idx,
                    frame_idx=idx,
                    pt=bootstrapper.pts2[i, 0:2],
                    des=bootstrapper.pts_des2[i],
                    transform=T,
                    landmark_id=i)
        else:
            # update keypoint trajectories
            prev_img = self.frame_queue[-1].img
            prev_keypoints, prev_trajectories, prev_landmarks = self.keypoint_trajectories.latest_keypoints(
            )
            print(
                f"landmarks: {len([l for l in prev_landmarks if l is not None])}")
            new_keypoints = np.array([kp.pt for kp in keypoints],
                                     dtype=np.float32)

            def min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                n_pts, _ = a.shape
                closest = np.zeros((n_pts), dtype=np.float32)
                for i in range(n_pts):
                    closest[i] = np.min(np.linalg.norm(a[i] - b, axis=1))
                return closest

            min_d = min_dist(new_keypoints, prev_keypoints)
            new_keypoints = new_keypoints[min_d > 20]

            to_track = np.vstack((prev_keypoints, new_keypoints))
            tracked_pts, status, err = cv.calcOpticalFlowPyrLK(
                prev_img, img, to_track, None, maxLevel=KLT_NUM_PYRAMIDS)

            # determine transformation
            n_prev_pts, _ = prev_keypoints.shape
            pts_mask = np.array([
                True if status[i] and prev_landmarks[i] is not None else False
                for i in range(n_prev_pts)
            ])
            img_pts = prev_keypoints[pts_mask]
            landmarks = np.array([prev_landmarks[i] for i, m in enumerate(pts_mask) if m ], dtype=np.float32)

            # Solve RANSAC P3P to extract rotation matrix and translation vector
            success, rvec, trans, inliers = cv.solvePnPRansac(
                landmarks,
                img_pts,
                self.K,
                distCoeffs=None,
                flags=cv.SOLVEPNP_ITERATIVE)
            assert success, "PNP RANSAC was not able to compute a pose from 2D - 3D correspondences"

            # Convert to homogeneous coordinates
            R, _ = cv.Rodrigues(rvec)
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = trans.ravel()

            # add previously tracked points
            for i in range(prev_keypoints.shape[0]):
                if not status[i]:
                    continue
                trajectory = prev_trajectories[i]
                self.keypoint_trajectories.tracked_to(trajectory.traj_idx, idx,
                                                      tracked_pts[i], None, T)
            # add newly tracked points
            for i in range(prev_keypoints.shape[0], tracked_pts.shape[0]):
                if not status[i]:
                    continue
                self.keypoint_trajectories.add_pt(idx, tracked_pts[i], None, T)

            # print(
            #     f"tracked {np.sum(status[:prev_keypoints.shape[0]])} old, {np.sum(status[prev_keypoints.shape[0]:])} new points to next frame"
            # )
            # print(
            #     f"mean trajectory length on frame {idx}: {self.keypoint_trajectories.mean_trajectory_length()}"
            # )

            # matched_pointcloud, img_kpts = self.poseEstimator.match_key_points(
            #     self.point_cloud, keypoints, descriptors,
            #     self.frame_queue[-1].img, self.frame_queue[-2].img)
            # M1 = self.poseEstimator.PnP(matched_pointcloud, img_kpts)

            # save img to frame queue
            self.frame_queue.append(FrameState(idx, img, T))

    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray,
                                 point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)
