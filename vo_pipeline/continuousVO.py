from typing import List, Tuple
from utils.matrix import hom_inv
from utils.frameQueue import FrameQueue
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.keypointTrajectory import KeypointTrajectories
from utils.loadData import Dataset
from vo_pipeline.frameState import FrameState
from scipy.spatial import KDTree
import cv2 as cv

import numpy as np


class ContinuousVO:
    def __init__(self,
                 dataset: Dataset,
                 descriptorType=ExtractorType.SIFT,
                 matcherType=MatcherType.BF,
                 useKLT=True,
                 algo_method=AlgoMethod.P3P,
                 max_point_distance: int = 50,
                 frames_to_skip: int = 4,
                 frame_queue_size: int = 100) -> None:

        self.dataset = dataset
        self.descriptor = FeatureExtractor(descriptorType)
        self.matcher = FeatureMatcher(matcherType)
        self.useKLT = useKLT
        self.poseEstimator = PoseEstimation(self.dataset.K,
                                            use_KLT=useKLT,
                                            algo_method_type=algo_method)

        # in-memory frame buffer
        assert frame_queue_size > 0
        self.frame_queue = FrameQueue(frame_queue_size)

        self.keyframes: List[FrameState] = []
        # camera calibration matrix
        self.K: np.ndarray = None
        # max point distance for bootstrapp algo
        self.max_point_distance: int = max_point_distance
        # number of frames to skip to get the first baseline
        self.frames_to_skip: int = frames_to_skip
        # save homogeneous coordinates of the last keyframe
        self.last_keyframe_coords: np.ndarray

        self.keypoint_trajectories = KeypointTrajectories()

        self.frame_idx = 0

    def step(self) -> None:
        """
        get the next frame and process it, i.e.
        - for the first n frame, just add them to frame queue
        - n-th frame pass to initialize point-cloud
        - after n-th frame pass them to _process_frame
        """
        K, img = next(self.dataset.frames)
        if self.frame_idx < self.frames_to_skip:
            self.K = K
            self.frame_queue.add(
                FrameState(self.frame_idx, img, np.eye(4, dtype=np.float32)))
        elif self.frame_idx == self.frames_to_skip:
            self._init_bootstrap(self.frame_idx, img)
        else:
            self._process_frame(self.frame_idx, img)

        self.frame_idx += 1

    def _add_keyframe(self, frame_state: FrameState):
        self.keyframes.append(frame_state)

    def _init_bootstrap(self, idx: int, img: np.ndarray) -> None:
        """
        init point-cloud and trajectories of those key-points found in the n-th image
        """
        # bootstrap initialization
        baseline = self.frame_queue.get(self.frames_to_skip - 2)
        T = self._bootstrap(baseline, idx, img)
        frame_state = FrameState(idx, img, T)
        self.frame_queue.add(frame_state)
        self._add_keyframe(frame_state)

    def _bootstrap(self, baseline: FrameState, frame_idx: int,
                   img: np.ndarray) -> np.ndarray:
        bootstrap = BootstrapInitializer(
            baseline.img, img, self.K, max_point_dist=self.max_point_distance)
        num_pts, _ = bootstrap.point_cloud.shape
        T = bootstrap.T @ baseline.pose
        new_landmarks = (hom_inv(baseline.pose) @ bootstrap.point_cloud.T).T

        for i in range(num_pts):
            landmark_id = len(self.keypoint_trajectories.landmarks)
            self.keypoint_trajectories.landmarks.append(new_landmarks[i, 0:3])
            trajectory, _ = self.keypoint_trajectories.create_trajectory(
                baseline.idx, bootstrap.pts1[i, 0:2], baseline.pose)
            self.keypoint_trajectories.tracked_to(traj_idx=trajectory.traj_idx,
                                                  frame_idx=frame_idx,
                                                  pt=bootstrap.pts2[i, 0:2],
                                                  transform=T,
                                                  landmark_id=landmark_id)
        return T

    def _optimal_baseline(self) -> Tuple[FrameState, int]:
        pass

    def _process_frame(self, frame_idx: int, img: np.ndarray) -> None:
        """
        given pointcloud is used to determine the trajectories of the keypoints in the frames
        """
        # get keypoints of the previous frame
        prev_img = self.frame_queue.get_head().img
        prev_keypoints, prev_trajectories, prev_landmarks = self.keypoint_trajectories.at_frame(
            self.keypoint_trajectories.latest_frame)

        #  prev_keypoints are tracked, the new keypoints are just important to init the trajectories later
        tracked_pts, status = PoseEstimation.KLT(prev_img, img, prev_keypoints)
        status = status.ravel() == 1
        # filter all previous keypoints that could not have been tracked in the image and remove the
        # corresponding landmarks
        tracked_pts = tracked_pts[status]
        tracked_landmarks = prev_landmarks[status]

        # Solve RANSAC P3P to extract rotation matrix and translation vector
        T, inliers = self.poseEstimator.PnP(tracked_landmarks, tracked_pts)
        inlier_ratio = inliers.shape[0] / tracked_pts.shape[0]

        # add tracked points
        trajectories = prev_trajectories[status]
        for i, tracked_pt in enumerate(tracked_pts):
            traj_idx = trajectories[i]
            self.keypoint_trajectories.tracked_to(traj_idx, frame_idx,
                                                  tracked_pt, T)

        # sample new keypoints
        new_keypoints = FeatureExtractor.harris(img)
        # for pt in new_keypoints:
        #     self.keypoint_trajectories.create_trajectory(frame_idx, pt, T)

        if max(0, frame_idx - self.frames_to_skip) % 10 == 0:
            baseline = self.frame_queue.get(4)
            self._bootstrap(baseline, frame_idx, img)

        print(
            f"{frame_idx}: tracked_pts: {tracked_pts.shape[0]:>5}, , \tinlier_ratio: {inlier_ratio:.2f}, \tadded_pts: {new_keypoints.shape[0]:>5}, mean traj len: {self.keypoint_trajectories.mean_trajectory_length():.2f}"
        )

        # save img to frame queue
        self.frame_queue.add(FrameState(frame_idx, img, T))

    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray,
                                 point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)
