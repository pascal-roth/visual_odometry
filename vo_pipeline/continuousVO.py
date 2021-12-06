from typing import List
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.keypointTrajectory import KeypointTrajectories
from utils.loadData import Dataset
from vo_pipeline.frameState import FrameState
from scipy.spatial import KDTree

import numpy as np


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
            self.frame_queue.append(FrameState(self.frame_idx, img, np.eye(4)))
        elif self.frame_idx == self.frames_to_skip:
            self._init_trajectories(self.frame_idx, img)
        else:
            self._process_frame(self.frame_idx, img)

        self.frame_idx += 1

    def _init_trajectories(self, idx: int, img: np.ndarray) -> None:
        """
        init point-cloud and trajectories of those key-points found in the n-th image
        """
        # bootstrap initialization
        img_init = self.frame_queue[0].img
        bootstrapper = BootstrapInitializer(img_init, img, self.K, max_point_dist=self.max_point_distance)

        # set the point-cloud in the keypoint trajectory class (= only place where point-cloud is saved)
        self.keypoint_trajectories.set_point_cloud(list(bootstrapper.point_cloud[:, 0:3]))
        T = bootstrapper.T

        # save img to frame queue
        self.frame_queue.append(FrameState(idx, img, T))

        # initialize keypoint trajectories
        for i in range(bootstrapper.pts1.shape[0]):
            trajectory = self.keypoint_trajectories.add_pt(0, bootstrapper.pts1[i, 0:2], bootstrapper.pts_des1[i],
                                                           np.eye(4))
            self.keypoint_trajectories.tracked_to(traj_idx=trajectory.traj_idx,
                                                  frame_idx=idx,
                                                  pt=bootstrapper.pts2[i, 0:2],
                                                  des=bootstrapper.pts_des2[i],
                                                  transform=T,
                                                  landmark_id=i)

    def _process_frame(self, idx: int, img: np.ndarray) -> None:
        """
        given pointcloud is used to determine the trajectories of the keypoints in the frames
        """
        # get keypoints of the previous frame
        prev_img = self.frame_queue[-1].img
        prev_keypoints, prev_trajectories, prev_landmarks = self.keypoint_trajectories.latest_keypoints()
        print(f"landmarks: {len([l for l in prev_landmarks if l is not None])}")

        # obtain SIFT keypoints and descriptors of the current frame
        keypoints, descriptors = self.descriptor.get_kp(img)
        new_keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # TODO: think about the method, maybe its best to match all keypoints and then discard the keypoints where
        #  second match is closer than a certain distance and not discarge in general all features too close to
        #  the keypoint in the previous frame
        kpts_kd_tree = KDTree(prev_keypoints)
        min_d, _ = kpts_kd_tree.query(new_keypoints)
        new_keypoints = new_keypoints[min_d > 20]  # TODO: tunable parameter

        # TODO: why should be the new keypoints which you have found in img be tracked in img ??? I would say that just
        #  prev_keypoints are tracked, the new keypoints are just important to init the trajectories later
        to_track = np.vstack((prev_keypoints, new_keypoints))
        tracked_pts, status = PoseEstimation.KLT(prev_img, img, to_track)

        # filter all previous keypoints that could not have been tracked in the image and remove the
        # corresponding landmarks
        n_prev_pts, _ = prev_keypoints.shape
        pts_mask = np.array([
            True if status[i] and prev_landmarks[i] is not None else False
            for i in range(n_prev_pts)
        ])
        img_pts = prev_keypoints[pts_mask]
        landmarks = np.array([prev_landmarks[i] for i, m in enumerate(pts_mask) if m], dtype=np.float32)

        # TODO: why here P3P RANSAC with prev_keypoints and not the ones of the current image ?
        # Solve RANSAC P3P to extract rotation matrix and translation vector
        T, inliers = self.poseEstimator.PnP(landmarks, img_pts)

        # add previously tracked points
        for i in range(prev_keypoints.shape[0]):
            if not status[i] and i in inliers:  # only use inliers
                continue
            trajectory = prev_trajectories[i]
            self.keypoint_trajectories.tracked_to(trajectory.traj_idx, idx,
                                                  tracked_pts[i], None, T)
        # add newly tracked points
        for i in range(prev_keypoints.shape[0], tracked_pts.shape[0]):
            if not status[i]:
                continue
            self.keypoint_trajectories.add_pt(idx, tracked_pts[i], None, T)

        # save img to frame queue
        self.frame_queue.append(FrameState(idx, img, T))

    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray,
                                 point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)
