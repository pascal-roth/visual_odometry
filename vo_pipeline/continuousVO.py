from typing import List, Tuple, Dict
from utils.matrix import hom_inv, to_hom
from utils.frameQueue import FrameQueue
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.keypointTrajectory import KeypointTrajectories
from utils.loadData import Dataset
from vo_pipeline.frameState import FrameState
from vo_pipeline.bundleAdjustment import BundleAdjustment
from params import *
import cv2 as cv
import warnings

import numpy as np


class ContinuousVO:
    def __init__(self,
                 dataset: Dataset,
                 descriptorType=ExtractorType.SIFT,
                 matcherType=MatcherType.BF,
                 useKLT=True,
                 algo_method=AlgoMethod.P3P,
                 max_point_distance: int = 100,
                 frames_to_skip: int = 10,
                 frame_queue_size: int = 1000) -> None:

        self.dataset = dataset
        self.descriptor = FeatureExtractor(descriptorType)
        self.matcher = FeatureMatcher(matcherType)
        self.useKLT = useKLT
        self.poseEstimator = PoseEstimation(self.dataset.K,
                                            use_KLT=useKLT,
                                            algo_method_type=algo_method)
        self.bundle_adjustment = BundleAdjustment(self.dataset.K)

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
        self.bootstrap_idx: List[int] = []
        self.inlier_count = 0

    def step(self) -> FrameState:
        """
        get the next frame and process it, i.e.
        - for the first n frame, just add them to frame queue
        - n-th frame pass to initialize point-cloud
        - after n-th frame pass them to _process_frame
        :return: index of last reconstructed frame
        """
        try:
            K, img = next(self.dataset.frames)
        except StopIteration:
            return None

        if self.frame_idx < self.frames_to_skip:
            self.K = K
            self.frame_queue.add(
                FrameState(self.frame_idx, img, np.eye(4, dtype=np.float32)))
        elif self.frame_idx == self.frames_to_skip:
            self._init_bootstrap(self.frame_idx, img)
        else:
            self._process_frame(self.frame_idx, img)
        self.frame_idx += 1
        return self.frame_queue.get_head()

    def _add_keyframe(self, frame_state: FrameState):
        self.keyframes.append(frame_state)

    def _init_bootstrap(self, idx: int, img: np.ndarray) -> None:
        """
        init point-cloud and trajectories of those key-points found in the n-th image
        """
        # bootstrap initialization
        baseline = self.frame_queue.get(self.frames_to_skip - 1)
        # img2 = baseline.img
        # baseline.img = img
        # T = self._bootstrap(baseline, idx, img2)
        T = self._bootstrap(baseline, idx, img)
        frame_state = FrameState(idx, img, T, is_key=True)
        self.frame_queue.add(frame_state)
        self._add_keyframe(frame_state)

    def _bootstrap(self,
                   baseline: FrameState,
                   frame_idx: int,
                   img: np.ndarray,
                   world_transform: np.ndarray = None) -> np.ndarray:
        """

        :param baseline:
        :param frame_idx:
        :param img:
        :param world_transform:
        :return: transform from baseline to img
        """
        # if frame_idx in self.keypoint_trajectories.on_frame:
        #     del self.keypoint_trajectories.on_frame[frame_idx]

        bootstrap = BootstrapInitializer(
            baseline.img, img, self.K, max_point_dist=self.max_point_distance)
        num_pts, _ = bootstrap.point_cloud.shape

        T = bootstrap.T @ baseline.pose
        # if not (baseline.pose == np.identity(4)).all():  # TODO: remove, just to prove a point
        #     T = baseline.pose
        # else:
        #     T = bootstrap.T @ baseline.pose

        bootstrap_scale = np.linalg.norm(T[0:3, 3])

        # if world_transform is None:
        #     # rescale T & landmarks to world scale
        #     rescaling_factor = 1 / bootstrap_scale
        #     T[0:3, 3] *= abs(rescaling_factor)
        # else:
        #     world_scale = np.linalg.norm(world_transform[0:3, 3])
        #     rescaling_factor = world_scale / bootstrap_scale
        #     T[0:3, 3] *= abs(rescaling_factor)
        # print(rescaling_factor)

        # transform landmarks to world frame
        landmarks = bootstrap.point_cloud
        new_landmarks = (hom_inv(baseline.pose) @ landmarks.T).T

        # initialize new trajectories
        self.inlier_count = num_pts
        for i in range(num_pts):
            landmark_id = len(self.keypoint_trajectories.landmarks)
            self.keypoint_trajectories.landmarks.append(new_landmarks[i, 0:3])
            trajectory, _ = self.keypoint_trajectories.create_trajectory(
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
        inliers = None
        if tracked_landmarks.shape[0] > 5:
            T, inliers = self.poseEstimator.PnP(tracked_landmarks, tracked_pts)
            inlier_ratio = inliers.shape[0] / self.inlier_count
        else:
            warnings.warn("too few points, forced bootstrap")
            inlier_ratio = 1
            prev_keyframe = self.keyframes[-1]
            T = self._bootstrap(
                prev_keyframe,
                frame_idx,
                img,
                world_transform=self.frame_queue.queue[-1].pose)
            self.bootstrap_idx.append(frame_idx)
            frame_state = FrameState(frame_idx,
                                     img,
                                     T,
                                     tracked_kps=tracked_pts.shape[0],
                                     is_key=True)
            self.frame_queue.add(frame_state)
            self.keyframes.append(frame_state)
            return

        # add tracked points
        inliers = set(inliers.ravel()) if inliers is not None else None
        trajectories = prev_trajectories[status]
        for i, tracked_pt in enumerate(tracked_pts):
            if inliers is None or i in inliers:
                traj_idx = trajectories[i]
                self.keypoint_trajectories.tracked_to(traj_idx, frame_idx,
                                                      tracked_pt, T)

        # if abs(T[0, 3] > 1.5 * abs(self.frame_queue.get(0).pose[0, 3])):
        #     raise AssertionError('Locally not consistent')

        is_key = False
        try:
            prev_keyframe = self.keyframes[-2]
        except IndexError:
            prev_keyframe = self.keyframes[-1]

        baseline_uncertainty = self._baseline_uncertainty(
            prev_keyframe.pose, T, tracked_landmarks)
        print(
            f"{frame_idx}: tracked_pts: {tracked_pts.shape[0]:>5}, inlier_ratio: {inlier_ratio:.2f}, baseline uncertainty: {baseline_uncertainty:.2f}"
        )
        if baseline_uncertainty > MAX_BASELINE_UNCERTAINTY or inlier_ratio < MIN_INLIER_RATIO:
            is_key = True
            T_bundle_adjustment = self._bundle_adjustment(frame_idx, T)

            # choose prev keyframe that is far away enough
            prev_idx = min(prev_keyframe.idx, frame_idx - MIN_FRAME_DIST)
            print(
                f"choosing prev_frame: {prev_idx}, prev_keyframe idx: {prev_keyframe.idx}"
            )
            # prev_keyframe = self.frame_queue.get(frame_idx - prev_idx - 1)
            prev_keyframe = self.frame_queue.get(2)
            T = self._bootstrap(
                prev_keyframe,
                frame_idx,
                img,
                world_transform=prev_keyframe.pose)  #T_bundle_adjustment)
            self.bootstrap_idx.append(frame_idx)

        # save img to frame queue
        frame_state = FrameState(frame_idx,
                                 img,
                                 T,
                                 tracked_kps=tracked_pts.shape[0],
                                 is_key=is_key)
        self.frame_queue.add(frame_state)
        if is_key:
            self.keyframes.append(frame_state)

    def _baseline_uncertainty(self, T0: np.ndarray, T1: np.ndarray,
                              landmarks: np.ndarray) -> float:
        T0_inv = hom_inv(T0)
        T1_inv = hom_inv(T1)

        # depth of the landmarks in first camera
        camera_normal = T0[0:3, 0:3].T @ np.array([[0], [0], [1]])
        camera_origin = T0_inv @ np.array([[0], [0], [0], [1]])
        centered_landmarks = landmarks - camera_origin[0:3].ravel()
        depths = []
        for landmark in centered_landmarks:
            d = np.dot(landmark, camera_normal.ravel())
            if d > 0:
                depths.append(d)

        if len(depths) == 0:
            return np.inf
        depth = np.mean(depths)
        # distance of the two poses
        init = T0_inv[:3, 3]
        final = T1_inv[:3, 3]
        dist = np.linalg.norm(final - init)
        return float(dist / depth)

    def _bundle_adjustment(self, frame_idx: int, T: np.ndarray):
        poses: List[np.ndarray] = []
        landmarks: List[np.ndarray] = []
        prev_keyframe = self.keyframes[max(
            -BUNDLE_ADJUSTMENT_KEYFRAME_LOOK_BACK, -len(self.keyframes))]
        look_back = frame_idx - prev_keyframe.idx
        if look_back >= self.frame_queue.size:
            look_back = self.frame_queue.size - 1
        keypoints: List[np.ndarray] = []
        point_indices: List[int] = []
        camera_indices: List[int] = []
        traj2landmark: Dict[int, int] = dict()

        # fill pose_graph optimization lists
        for i in range(look_back + 1):
            old_frame_idx = frame_idx - look_back + i
            if old_frame_idx == frame_idx:
                pose = T
            else:
                frame_state = self.frame_queue.get(look_back - i - 1)
                pose = frame_state.pose
            poses.append(pose[0:3, 0:4])
            kp, traj, lnd = self.keypoint_trajectories.at_frame(old_frame_idx)

            keypoints += list(kp)
            for traj_idx in traj:
                if traj_idx not in traj2landmark:
                    landmark = self.keypoint_trajectories.landmarks[
                        self.keypoint_trajectories.traj2landmark[traj_idx]]
                    traj2landmark[traj_idx] = len(landmarks)
                    landmarks.append(landmark)

                point_indices.append(traj2landmark[traj_idx])
                camera_indices.append(i)

        # create numpy arrays
        poses_np = np.asarray(poses, dtype=np.float32)
        landmarks_np = np.asarray(landmarks, dtype=np.float32)
        keypoints_np = np.asarray(keypoints, dtype=np.float32)
        camera_indices_np = np.asarray(camera_indices)
        point_indices_np = np.asarray(point_indices)

        # run least squares optimization
        adjusted_poses, adjusted_landmarks = self.bundle_adjustment.bundle_adjustment(
            poses_np, landmarks_np, camera_indices_np, point_indices_np,
            keypoints_np)
        adjusted_poses_hom = [
            np.vstack((pose, np.array([[0, 0, 0, 1]], dtype=np.float32)))
            for pose in adjusted_poses
        ]

        # move adjusted poses back to prev_keyframe transform
        # first_pose = adjusted_poses_hom[0]
        # transform_error = prev_keyframe.pose @ hom_inv(first_pose)
        # transform_adj = hom_inv(transform_error)
        # adjusted_poses_hom = [transform_adj @ pose for pose in adjusted_poses_hom]

        updated = []
        # update poses for already tracked frames
        for i in range(look_back):
            frame_state = self.frame_queue.get(look_back - i - 1)
            frame_state.pose = np.copy(adjusted_poses_hom[i])
            updated.append(frame_state.idx)

        # update landmarks
        landmark2traj = {v: k for k, v in traj2landmark.items()}
        for i, landmark in enumerate(adjusted_landmarks):
            traj_idx = landmark2traj[i]
            landmark_idx = self.keypoint_trajectories.traj2landmark[traj_idx]
            self.keypoint_trajectories.landmarks[landmark_idx] = landmark

        return adjusted_poses_hom[-1]

    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray,
                                 point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)
