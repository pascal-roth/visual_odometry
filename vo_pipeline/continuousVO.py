from typing import Tuple
from main import poseEstimation_example
from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from bootstrap import BootstrapInitializer
from poseEstimation import PoseEstimation

import numpy as np


class continuousVO:

    def __init__(self, datasetType = DatasetType.KITTI, descriptorType = ExtractorType.SIFT,
                matcherType = MatcherType.BF, useKLT = True, algo_method = AlgoMethod.P3P) -> None:

        self.dataset    = DatasetLoader(datasetType).load()
        self.descriptor = FeatureExtractor(descriptorType)
        self.matcher    = FeatureMatcher(MatcherType)

        self.poseEstimator = PoseEstimation(self.dataset.K, useKLT, algo_method)

    def runPipeline():
        pass

    def _processFrame(self, S0 : Tuple[np.ndarray], I0 : np.ndarray, I1 : np.ndarray):
        """
        :param S0:          dictionary state of last frame: contains last frame's 
                            'keypoints', 'landmarks', candidate keypoints 
                            (current pose and initial pose) and initial poses
        :param I0           past frame
        :param I1           current frame
        :return             tuple with state of the current frame and current pose
        """

        pointcloud  = S0.pointcloud
        keypoints   = S0.keypoints
        descriptors = S0.descriptors

        matched_pointcloud, img_kpts = self.poseEstimator.match_key_points(pointcloud, keypoints, descriptors, I1, I0)
        M1 = self.poseEstimator.PnP(matched_pointcloud, img_kpts)
        # Maybe np.linalg.inv(M[idx])

        return S1, M1

    def _addNewLandmarks():
        pass

class FrameState:

    keypoints       : np.ndarray        # 2 x N matrix 
    descriptors     : np.ndarray        # 16x N matrix
    pointcloud      : np.ndarray        # 3 x N matrix
    candidate_C     : np.ndarray        # 2 x M matrix
    candidate_F     : np.ndarray        # 2 x M matrix
    initial_poses   : Tuple[np.ndarray] # 12x M matrix
    