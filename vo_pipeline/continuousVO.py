from typing import Tuple, List, Dict
from main import poseEstimation_example
from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from bootstrap import BootstrapInitializer
from poseEstimation import PoseEstimation
from utils.loadData import Dataset
from queue import Queue
from frameState import FrameState
from params import * 

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
        self.poseEstimator = PoseEstimation(self.dataset.K, useKLT, algo_method)

        # in-memory frame buffer 
        self.frame_queue: List[FrameState] = []
        self.point_cloud: List[np.ndarray] = []
        # camera calibration matrix
        self.K: np.ndarray = None
        # max point distance for bootstrapp algo
        self.max_point_distance: int = max_point_distance
        # number of frames to skip to get the first baseline
        self.frames_to_skip: int = frames_to_skip
        # save homogeneous coordinates of the last keyframe
        self.last_keyframe_coords: np.ndarray 

        self.keypoint_trajectories: Dict[int, KeypointTrajectory] = dict()

    def run_pipeline(self):
        for i, D in enumerate(self.dataset.frames):
            K, img = D
            if i <= self.frames_to_skip:
                self.K = K
                self.frame_queue.append(FrameState(i, img, np.eye(4)))
            else:
                self._processFrame()


    def _processFrame(self):
        """
        :param S0:          dictionary state of last frame: contains last frame's 
                            'keypoints', 'landmarks', candidate keypoints 
                            (current pose and initial pose) and initial poses
        :param I0           past frame
        :param I1           current frame
        :return             tuple with state of the current frame and current pose
        """
    
        kpts = self.descriptor.get_kp(self.frame_queue[-1].img)
        self.frame_queue[-1].keypoints = kposeEstimation_example
        if not self.point_cloud:
            img1 = self.frame_queue[-1].img
            img2 = self.frame_queue[0].img
            bootstrapper = BootstrapInitializer(img1, img2, self.K, max_point_dist=self.max_point_distance) 
            self.last_keyframe_coords =  bootstrapper.pts2[:, 0:2]
            self.point_cloud = bootstrapper.point_cloud      
        else:
            poseEstimator = PoseEstimation(self.K, self.point_cloud[:,0:3], self.last_keyframe_coords, use_KLT=self.useKLT, algo_method_type=AlgoMethod.P3P)
        
            
            

        matched_pointcloud, img_kpts = self.poseEstimator.match_key_points(
            pointcloud, keypoints, descriptors, I1, I0)
        M1 = self.poseEstimator.PnP(matched_pointcloud, img_kpts)
        # Maybe np.linalg.inv(M[idx])

    def _addNewLandmarks():
        pass

    @staticmethod
    def get_baseline_uncertainty(T: np.ndarray, point_cloud: np.ndarray) -> float:
        depths = point_cloud[:, 2]
        mean_depth = np.mean(depths)
        key_dist = np.linalg.norm(T[0:3, 3])
        return float(key_dist / mean_depth)
        

class KeypointTrajectory:
    def __init__(self):
        pass