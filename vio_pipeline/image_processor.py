import time
from collections import defaultdict, namedtuple
from itertools import chain
from typing import List

import cv2
import numpy as np
from params import *
from utils.message import FeatureData

from vio_pipeline.featureExtraction import FeatureExtractor
from vio_pipeline.poseEstimation import PoseEstimation

from utils.message import FeatureMeasurement, FeatureMetaData


class ImageProcessor(object):
    """
    Detect and track features in image sequences.
    """
    def __init__(self, calibration_mat: np.ndarray, R_CAM_IMU: np.ndarray,
                 T_CAM_IMU: np.ndarray):
        # Indicate if this is the first image message.
        self.is_first_img = True

        # ID for the next new feature.
        self.next_feature_id = 0

        # Feature detector
        self.detector = FeatureExtractor()
        # IMU message buffer.
        self.imu_buffer = []

        # Previous and current images
        self.prev_frame = None
        self.curr_frame = None

        # Features in the previous and current image.
        # list of lists of FeatureMetaData
        self.prev_features = [[] for _ in range(GRID_NUM)]
        self.curr_features = [[] for _ in range(GRID_NUM)]

        # Number of features after each outlier removal step.
        # keys: before_tracking, after_tracking, after_matching, after_ransac
        self.num_features = defaultdict(int)

        # Camera calibration parameters
        self.K = calibration_mat  # vec4

        # Take a vector from cam0 frame to the IMU frame.
        self.R_CAM_IMU = R_CAM_IMU

    def mono_callback(self, curr_frame):
        """
        Callback function for the mono images.
        """
        self.curr_frame = curr_frame

        # Detect features in the first frame.
        if self.is_first_img:
            self.initialize_first_frame()
            self.is_first_img = False
        else:
            # Track the feature in the previous image.
            self.track_features()

            # Add new features into the current image.
            self.add_new_features()
            self.prune_features()

        try:
            return self.publish()
        finally:
            self.prev_frame = self.curr_frame
            self.prev_features = self.curr_features

            # Initialize the current features to empty vectors.
            self.curr_features = [[] for _ in range(GRID_NUM)]

    def imu_callback(self, msg):
        """
        Callback function for the imu message.
        """
        self.imu_buffer.append(msg)

    def initialize_first_frame(self):
        """
        Initialize the image processing sequence, which is basically detect
        new features on the first set of stereo images.
        """
        img = self.curr_frame.image
        grid_height, grid_width = self.get_grid_size(img)

        # Detect new features on the frist image.
        new_features, _ = self.detector.get_kp(img)

        # Find the stereo matched points for the newly detected features.
        kpts = [kp.pt for kp in new_features]
        kpts_response = [kp.response for kp in new_features]

        # Group the features into grids
        grid_new_features = [[] for _ in range(GRID_NUM)]

        for i in range(len(kpts)):
            cam0_point = kpts[i]
            response = kpts_response[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row * GRID_COL + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features,
                                  key=lambda x: x.response,
                                  reverse=True)[:GRID_MIN_FEATURE]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1

    def track_features(self):
        """
        Tracker features on the newly received stereo images.
        """
        img = self.curr_frame.image
        grid_height, grid_width = self.get_grid_size(img)

        # Compute a rough relative rotation which takes a vector
        # from the previous frame to the current frame.
        cam0_R_p_c = self.integrate_imu_data()

        # Organize the features in the previous image.
        prev_ids = []
        prev_lifetime = []
        prev_kpts = []

        for feature in chain.from_iterable(self.prev_features):
            prev_ids.append(feature.id)
            prev_lifetime.append(feature.lifetime)
            prev_kpts.append(feature.cam0_point)
        prev_kpts = np.array(prev_kpts, dtype=np.float32)

        # Number of the features before tracking.
        self.num_features['before_tracking'] = len(prev_kpts)

        # Abort tracking if there is no features in the previous frame.
        if len(prev_kpts) == 0:
            return

        # Track features using LK optical flow method.
        curr_kpts = self.predict_feature_tracking(prev_kpts, cam0_R_p_c)

        curr_kpts, track_inliers = PoseEstimation.KLT(
            self.prev_frame.image, self.curr_frame.image,
            prev_kpts.astype(np.float32), curr_kpts.astype(np.float32))

        # Mark those tracked points out of the image region as untracked.
        for i, point in enumerate(curr_kpts):
            if not track_inliers[i]:
                continue
            if (point[0] < 0 or point[0] > img.shape[1] - 1 or point[1] < 0
                    or point[1] > img.shape[0] - 1):
                track_inliers[i] = 0

        # Collect the tracked points.
        prev_tracked_ids = select(prev_ids, track_inliers)
        prev_tracked_lifetime = select(prev_lifetime, track_inliers)
        prev_tracked_kpts = select(prev_kpts, track_inliers)
        curr_tracked_kpts = select(curr_kpts, track_inliers)

        # Number of features left after tracking.
        self.num_features['after_tracking'] = len(curr_tracked_kpts)

        # # Number of features after ransac.
        after_ransac = 0
        for i in range(len(curr_tracked_kpts)):
            row = int(curr_tracked_kpts[i][1] / grid_height)
            col = int(curr_tracked_kpts[i][0] / grid_width)
            code = row * GRID_COL + col

            grid_new_feature = FeatureMetaData()
            grid_new_feature.id = prev_tracked_ids[i]
            grid_new_feature.lifetime = prev_tracked_lifetime[i] + 1
            grid_new_feature.cam0_point = curr_tracked_kpts[i]
            prev_tracked_lifetime[i] += 1

            self.curr_features[code].append(grid_new_feature)
            after_ransac += 1
        self.num_features['after_ransac'] = after_ransac

    def add_new_features(self):
        """
        Detect new features on the image to ensure that the features are
        uniformly distributed on the image.
        """
        curr_img = self.curr_frame.image
        grid_height, grid_width = self.get_grid_size(curr_img)

        # Create a mask to avoid redetecting existing features.
        mask = np.ones(curr_img.shape[:2], dtype='uint8')

        for feature in chain.from_iterable(self.curr_features):
            x, y = map(int, feature.cam0_point)
            mask[y - 3:y + 4, x - 3:x + 4] = 0

        # Detect new features.
        # new_features = self.detector.detect(curr_img, mask=mask)
        new_features, _ = self.detector.get_kp(curr_img, mask=mask)

        # Collect the new detected features based on the grid.
        # Select the ones with top response within each grid afterwards.
        new_feature_sieve = [[] for _ in range(GRID_NUM)]
        for feature in new_features:
            row = int(feature.pt[1] / grid_height)
            col = int(feature.pt[0] / grid_width)
            code = row * GRID_COL + col
            new_feature_sieve[code].append(feature)

        new_features = []
        for features in new_feature_sieve:
            if len(features) > GRID_MAX_FEATURE:
                features = sorted(features,
                                  key=lambda x: x.response,
                                  reverse=True)[:GRID_MAX_FEATURE]
            new_features.append(features)
        new_features = list(chain.from_iterable(new_features))

        # Find the stereo matched points for the newly detected features.
        kpts = [kp.pt for kp in new_features]
        kpts_response = [kp.response for kp in new_features]

        # Group the features into grids
        grid_new_features = [[] for _ in range(GRID_NUM)]
        for i in range(len(kpts)):
            cam0_point = kpts[i]
            response = kpts_response[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row * GRID_COL + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features,
                                  key=lambda x: x.response,
                                  reverse=True)[:GRID_MIN_FEATURE]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1

    def prune_features(self):
        """
        Remove some of the features of a grid in case there are too many
        features inside of that grid, which ensures the number of features
        within each grid is bounded.
        """
        for i, features in enumerate(self.curr_features):
            # Continue if the number of features in this grid does
            # not exceed the upper bound.
            if len(features) <= GRID_MAX_FEATURE:
                continue
            self.curr_features[i] = sorted(features,
                                           key=lambda x: x.lifetime,
                                           reverse=True)[:GRID_MAX_FEATURE]

    def publish(self) -> FeatureData:
        """
        Publish the features on the current image including both the
        tracked and newly detected ones.
        """
        curr_ids = []
        curr_kpts = []
        for feature in chain.from_iterable(self.curr_features):
            curr_ids.append(feature.id)
            curr_kpts.append(feature.cam0_point)

        features: List[FeatureMeasurement] = [None] * len(curr_ids)
        for i in range(len(curr_ids)):
            fm = FeatureMeasurement()
            fm.id = curr_ids[i]
            fm.u0 = curr_kpts[i][0]
            fm.v0 = curr_kpts[i][1]
            features.append(fm)

        return FeatureData(self.curr_frame.timestamp, features)

    def integrate_imu_data(self):
        """
        Integrates the IMU gyro readings between the two consecutive images,
        which is used for both tracking prediction and 2-point RANSAC.

        Returns:
            cam0_R_p_c: a rotation matrix which takes a vector from previous
                cam0 frame to current cam0 frame.
            cam1_R_p_c: a rotation matrix which takes a vector from previous
                cam1 frame to current cam1 frame.
        """
        # Find the start and the end limit within the imu msg buffer.
        idx_begin = None
        for i, msg in enumerate(self.imu_buffer):
            if msg.timestamp >= self.prev_frame.timestamp - 0.01:
                idx_begin = i
                break

        idx_end = None
        for i, msg in enumerate(self.imu_buffer):
            if msg.timestamp >= self.curr_frame.timestamp - 0.004:
                idx_end = i
                break

        if idx_begin is None or idx_end is None:
            return np.identity(3)

        # Compute the mean angular velocity in the IMU frame.
        mean_ang_vel = np.zeros(3)
        for i in range(idx_begin, idx_end):
            mean_ang_vel += self.imu_buffer[i].angular_velocity

        if idx_end > idx_begin:
            mean_ang_vel /= (idx_end - idx_begin)

        # Transform the mean angular velocity from the IMU frame to the
        # cam0 and cam1 frames.
        cam0_mean_ang_vel = self.R_CAM_IMU.T @ mean_ang_vel
        # cam1_mean_ang_vel = self.R_cam1_imu.T @ mean_ang_vel

        # Compute the relative rotation.
        dt = self.curr_frame.timestamp - self.prev_frame.timestamp
        cam0_R_p_c = cv2.Rodrigues(cam0_mean_ang_vel * dt)[0].T
        # cam1_R_p_c = cv2.Rodrigues(cam1_mean_ang_vel * dt)[0].T

        # Delete the useless and used imu messages.
        self.imu_buffer = self.imu_buffer[idx_end:]
        return cam0_R_p_c

    def rescale_points(self, pts1, pts2):
        """
        Arguments:
            pts1: first set of points.
            pts2: second set of points.

        Returns:
            pts1: scaled first set of points.
            pts2: scaled second set of points.
            scaling_factor: scaling factor
        """
        scaling_factor = 0
        for pt1, pt2 in zip(pts1, pts2):
            scaling_factor += np.linalg.norm(pt1)
            scaling_factor += np.linalg.norm(pt2)

        scaling_factor = (len(pts1) + len(pts2)) / scaling_factor * np.sqrt(2)

        for i in range(len(pts1)):
            pts1[i] *= scaling_factor
            pts2[i] *= scaling_factor

        return pts1, pts2, scaling_factor

    def get_grid_size(self, img):
        """
        # Size of each grid.
        """
        grid_height = int(np.ceil(img.shape[0] / GRID_ROW))
        grid_width = int(np.ceil(img.shape[1] / GRID_COL))
        return grid_height, grid_width

    def predict_feature_tracking(self, input_pts, R_p_c):
        """
        predictFeatureTracking Compensates the rotation between consecutive
        camera frames so that feature tracking would be more robust and fast.

        Arguments:
            input_pts: features in the previous image to be tracked.
            R_p_c: a rotation matrix takes a vector in the previous camera
                frame to the current camera frame. (matrix33)

        Returns:
            compensated_pts: predicted locations of the features in the
                current image based on the provided rotation.
        """
        # Return directly if there are no input features.
        if len(input_pts) == 0:
            return []

        # Intrinsic matrix.
        H = self.K @ R_p_c @ np.linalg.inv(self.K)

        compensated_pts = []
        for i in range(len(input_pts)):
            p1 = np.array([*input_pts[i], 1.0])
            p2 = H @ p1
            compensated_pts.append(p2[:2] / p2[2])
        return np.array(compensated_pts, dtype=np.float32)


def select(data, selectors):
    return [d for d, s in zip(data, selectors) if s]
