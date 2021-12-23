import numpy as np
import cv2
import time

from itertools import chain, compress
from collections import defaultdict, namedtuple

from vio_pipeline.featureExtraction import FeatureExtractor
from vio_pipeline.poseEstimation import PoseEstimation
from params import *


class FeatureMetaData(object):
    """
    Contain necessary information of a feature for easy access.
    """

    def __init__(self):
        self.id = None  # int
        self.response = None  # float
        self.lifetime = None  # int
        self.cam0_point = None  # vec2
        # self.cam1_point = None   # vec2


class FeatureMeasurement(object):
    """
    Stereo measurement of a feature.
    """

    def __init__(self):
        self.id = None
        self.u0 = None
        self.v0 = None
        self.u1 = None
        self.v1 = None


class ImageProcessor(object):
    """
    Detect and track features in image sequences.
    """

    def __init__(self, calibration_mat: np.ndarray,
                 R_CAM_IMU: np.ndarray,
                 T_CAM_IMU: np.ndarray):
        # Indicate if this is the first image message.
        self.is_first_img = True

        # ID for the next new feature.
        self.next_feature_id = 0

        # Feature detector
        # self.detector = cv2.FastFeatureDetector_create(self.config.fast_threshold)
        self.detector = FeatureExtractor()
        # IMU message buffer.
        self.imu_msg_buffer = []

        # Previous and current images
        self.prev_frame = None
        self.curr_frame = None
        # self.cam1_curr_img_msg = None

        # Pyramids for previous and current image
        # self.prev_cam0_pyramid = None
        # self.curr_cam0_pyramid = None
        # self.curr_cam1_pyramid = None

        # Features in the previous and current image.
        # list of lists of FeatureMetaData
        self.prev_features = [[] for _ in range(GRID_NUM)]  # Don't use [[]] * N
        self.curr_features = [[] for _ in range(GRID_NUM)]

        # Number of features after each outlier removal step.
        # keys: before_tracking, after_tracking, after_matching, after_ransac
        self.num_features = defaultdict(int)

        # load config
        # Camera calibration parameters
        # self.cam0_resolution = config.cam0_resolution  # vec2
        self.cam0_intrinsics = calibration_mat  # vec4
        # self.cam0_distortion_model = config.cam0_distortion_model     # string
        # self.cam0_distortion_coeffs = config.cam0_distortion_coeffs   # vec4

        # self.cam1_resolution = config.cam1_resolution   # vec2
        # self.cam1_intrinsics = config.cam1_intrinsics   # vec4
        # self.cam1_distortion_model = config.cam1_distortion_model     # string
        # self.cam1_distortion_coeffs = config.cam1_distortion_coeffs   # vec4

        # Take a vector from cam0 frame to the IMU frame.
        self.R_CAM_IMU = R_CAM_IMU
        self.t_cam0_imu = T_CAM_IMU
        # # Take a vector from cam1 frame to the IMU frame.
        # self.T_cam1_imu = np.linalg.inv(config.T_imu_cam1)
        # self.R_cam1_imu = self.T_cam1_imu[:3, :3]
        # self.t_cam1_imu = self.T_cam1_imu[:3, 3]

    def mono_callback(self, curr_frame):
        """
        Callback function for the stereo images.
        """
        start = time.time()
        self.curr_frame = curr_frame
        # self.cam1_curr_img_msg = stereo_msg.cam1_msg

        # Build the image pyramids once since they're used at multiple places.
        # self.curr_cam0_pyramid = self.curr_frame.image

        # Detect features in the first frame.
        if self.is_first_img:
            self.initialize_first_frame()
            self.is_first_img = False
            # Draw results.
            # self.draw_features_stereo()
        else:
            # Track the feature in the previous image.
            t = time.time()
            self.track_features()
            print('___track_features:', time.time() - t)
            t = time.time()

            # Add new features into the current image.
            self.add_new_features()
            print('___add_new_features:', time.time() - t)
            t = time.time()
            self.prune_features()
            print('___prune_features:', time.time() - t)
            t = time.time()
            # Draw results.
            # self.draw_features_stereo()
            print('___draw_features_stereo:', time.time() - t)
            t = time.time()

        print('===image process elapsed:', time.time() - start, f'({curr_frame.timestamp})')

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
        self.imu_msg_buffer.append(msg)

    # def create_image_pyramids(self):
    #     """
    #     Create image pyramids used for KLT tracking.
    #     (Seems doesn't work in python)
    #     """
    #     curr_cam0_img = self.curr_frame.image
    #     # self.curr_cam0_pyramid = cv2.buildOpticalFlowPyramid(
    #     #     curr_cam0_img, self.config.win_size, self.config.pyramid_levels,
    #     #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False)[1]
    #     self.curr_cam0_pyramid = curr_cam0_img
    #
    #     # curr_cam1_img = self.cam1_curr_img_msg.image
    #     # # self.curr_cam1_pyramid = cv2.buildOpticalFlowPyramid(
    #     # #     curr_cam1_img, self.config.win_size, self.config.pyramid_levels,
    #     # #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False)[1]
    #     # self.curr_cam1_pyramid = curr_cam1_img

    def initialize_first_frame(self):
        """
        Initialize the image processing sequence, which is basically detect
        new features on the first set of stereo images.
        """
        img = self.curr_frame.image
        grid_height, grid_width = self.get_grid_size(img)

        # Detect new features on the frist image.
        # new_features = self.detector.detect(img)
        new_features, _ = self.detector.get_kp(img)

        # Find the stereo matched points for the newly detected features.
        kpts = [kp.pt for kp in new_features]
        # cam1_points, inlier_markers = self.stereo_match(kpts)

        cam0_inliers = kpts
        response_inliers = [kp.response for kp in new_features]
        # cam0_inliers, cam1_inliers = [], []
        # response_inliers = []
        # for i, inlier in enumerate(inlier_markers):
        #     if not inlier:
        #         continue
        #     cam0_inliers.append(kpts[i])
        #     cam1_inliers.append(cam1_points[i])
        #     response_inliers.append(new_features[i].response)
        # len(cam0_inliers) < max(5, 0.1 * len(new_features))

        # Group the features into grids
        grid_new_features = [[] for _ in range(GRID_NUM)]

        for i in range(len(cam0_inliers)):
            cam0_point = cam0_inliers[i]
            # cam1_point = cam1_inliers[i]
            response = response_inliers[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row * GRID_COL + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            # new_feature.cam1_point = cam1_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x: x.response,
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
        # cam0_R_p_c, cam1_R_p_c = self.integrate_imu_data()
        cam0_R_p_c = self.integrate_imu_data()

        # Organize the features in the previous image.
        prev_ids = []
        prev_lifetime = []
        prev_kpts = []
        # prev_cam1_points = []

        for feature in chain.from_iterable(self.prev_features):
            prev_ids.append(feature.id)
            prev_lifetime.append(feature.lifetime)
            prev_kpts.append(feature.cam0_point)
            # prev_cam1_points.append(feature.cam1_point)
        prev_kpts = np.array(prev_kpts, dtype=np.float32)

        # Number of the features before tracking.
        self.num_features['before_tracking'] = len(prev_kpts)

        # Abort tracking if there is no features in the previous frame.
        if len(prev_kpts) == 0:
            return

        # Track features using LK optical flow method.
        curr_kpts = self.predict_feature_tracking(
            prev_kpts, cam0_R_p_c, self.cam0_intrinsics)

        curr_kpts, track_inliers = PoseEstimation.KLT(self.prev_frame.image, self.curr_frame.image,
                                                      prev_kpts.astype(np.float32), curr_kpts.astype(np.float32))

        # Mark those tracked points out of the image region as untracked.
        for i, point in enumerate(curr_kpts):
            if not track_inliers[i]:
                continue
            if (point[0] < 0 or point[0] > img.shape[1] - 1 or
                    point[1] < 0 or point[1] > img.shape[0] - 1):
                track_inliers[i] = 0

        # Collect the tracked points.
        prev_tracked_ids = select(prev_ids, track_inliers)
        prev_tracked_lifetime = select(prev_lifetime, track_inliers)
        prev_tracked_kpts = select(prev_kpts, track_inliers)
        # prev_tracked_cam1_points = select(prev_cam1_points, track_inliers)
        curr_tracked_kpts = select(curr_kpts, track_inliers)

        # Number of features left after tracking.
        self.num_features['after_tracking'] = len(curr_tracked_kpts)

        # Outlier removal involves three steps, which forms a close
        # loop between the previous and current frames of cam0 (left)
        # and cam1 (right). Assuming the stereo matching between the
        # previous cam0 and cam1 images are correct, the three steps are:
        #
        # prev frames cam0 ----------> cam1
        #              |                |
        #              |ransac          |ransac
        #              |   stereo match |
        # curr frames cam0 ----------> cam1
        #
        # 1) Stereo matching between current images of cam0 and cam1.
        # 2) RANSAC between previous and current images of cam0.
        # 3) RANSAC between previous and current images of cam1.
        #
        # For Step 3, tracking between the images is no longer needed.
        # The stereo matching results are directly used in the RANSAC.

        # # Step 1: stereo matching.
        # curr_cam1_points, match_inliers = self.stereo_match(
        #     curr_tracked_kpts)
        #
        # prev_matched_ids = select(prev_tracked_ids, match_inliers)
        # prev_matched_lifetime = select(prev_tracked_lifetime, match_inliers)
        # prev_matched_kpts = select(prev_tracked_kpts, match_inliers)
        # prev_matched_cam1_points = select(prev_tracked_cam1_points, match_inliers)
        # curr_matched_kpts = select(curr_tracked_kpts, match_inliers)
        # curr_matched_cam1_points = select(curr_cam1_points, match_inliers)
        #
        # # Number of features left after stereo matching.
        # self.num_features['after_matching'] = len(curr_matched_kpts)
        #
        # # Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
        # # cam0_ransac_inliers = self.two_point_ransac(
        # #     prev_matched_kpts, curr_matched_kpts,
        # #     cam0_R_p_c, self.cam0_intrinsics,
        # #     self.cam0_distortion_model, self.cam0_distortion_coeffs,
        # #     self.config.ransac_threshold, 0.99)
        #
        # # cam1_ransac_inliers = self.two_point_ransac(
        # #     prev_matched_cam1_points, curr_matched_cam1_points,
        # #     cam1_R_p_c, self.cam1_intrinsics,
        # #     self.cam1_distortion_model, self.cam1_distortion_coeffs,
        # #     self.config.ransac_threshold, 0.99)
        # cam0_ransac_inliers = [1] * len(prev_matched_kpts)
        # cam1_ransac_inliers = [1] * len(prev_matched_cam1_points)
        #
        # # Number of features after ransac.
        after_ransac = 0
        for i in range(len(curr_tracked_kpts)):
            # if not (cam0_ransac_inliers[i] and cam1_ransac_inliers[i]):
            #     continue
            row = int(curr_tracked_kpts[i][1] / grid_height)
            col = int(curr_tracked_kpts[i][0] / grid_width)
            code = row * GRID_COL + col

            grid_new_feature = FeatureMetaData()
            grid_new_feature.id = prev_tracked_ids[i]
            grid_new_feature.lifetime = prev_tracked_lifetime[i] + 1
            grid_new_feature.cam0_point = curr_tracked_kpts[i]
            # grid_new_feature.cam1_point = curr_matched_cam1_points[i]
            prev_tracked_lifetime[i] += 1

            self.curr_features[code].append(grid_new_feature)
            after_ransac += 1
        self.num_features['after_ransac'] = after_ransac

        # Compute the tracking rate.
        # prev_feature_num = sum([len(x) for x in self.prev_features])
        # curr_feature_num = sum([len(x) for x in self.curr_features])

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
                features = sorted(features, key=lambda x: x.response,
                                  reverse=True)[:GRID_MAX_FEATURE]
            new_features.append(features)
        new_features = list(chain.from_iterable(new_features))

        # Find the stereo matched points for the newly detected features.
        kpts = [kp.pt for kp in new_features]
        response_inliers = [kp.response for kp in new_features]
        # cam1_points, inlier_markers = self.stereo_match(kpts)

        # cam0_inliers, cam1_inliers, response_inliers = [], [], []
        # for i, inlier in enumerate(inlier_markers):
        #     if not inlier:
        #         continue
        #     cam0_inliers.append(kpts[i])
        #     cam1_inliers.append(cam1_points[i])
        #     response_inliers.append(new_features[i].response)
        # if len(cam0_inliers) < max(5, len(new_features) * 0.1):

        # Group the features into grids
        cam0_inliers = kpts
        grid_new_features = [[] for _ in range(GRID_NUM)]
        for i in range(len(cam0_inliers)):
            cam0_point = cam0_inliers[i]
            # cam1_point = cam1_inliers[i]
            response = response_inliers[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row * GRID_COL + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            # new_feature.cam1_point = cam1_point
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x: x.response,
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
            self.curr_features[i] = sorted(features, key=lambda x: x.lifetime,
                                           reverse=True)[:GRID_MAX_FEATURE]

    def publish(self):
        """
        Publish the features on the current image including both the
        tracked and newly detected ones.
        """
        curr_ids = []
        curr_kpts = []
        # curr_cam1_points = []
        for feature in chain.from_iterable(self.curr_features):
            curr_ids.append(feature.id)
            curr_kpts.append(feature.cam0_point)
            # curr_cam1_points.append(feature.cam1_point)

        # curr_kpts_undistorted = self.undistort_points(
        #     curr_kpts, self.cam0_intrinsics,
        #     self.cam0_distortion_model, self.cam0_distortion_coeffs)
        # curr_cam1_points_undistorted = self.undistort_points(
        #     curr_cam1_points, self.cam1_intrinsics,
        #     self.cam1_distortion_model, self.cam1_distortion_coeffs)

        features = []
        for i in range(len(curr_ids)):
            fm = FeatureMeasurement()
            fm.id = curr_ids[i]
            fm.u0 = curr_kpts[i][0]
            fm.v0 = curr_kpts[i][1]
            # fm.u1 = curr_cam1_points_undistorted[i][0]
            # fm.v1 = curr_cam1_points_undistorted[i][1]
            features.append(fm)

        feature_msg = namedtuple('feature_msg', ['timestamp', 'features'])(
            self.curr_frame.timestamp, features)
        return feature_msg

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
        for i, msg in enumerate(self.imu_msg_buffer):
            if msg.timestamp >= self.prev_frame.timestamp - 0.01:
                idx_begin = i
                break

        idx_end = None
        for i, msg in enumerate(self.imu_msg_buffer):
            if msg.timestamp >= self.curr_frame.timestamp - 0.004:
                idx_end = i
                break

        if idx_begin is None or idx_end is None:
            return np.identity(3), np.identity(3)

        # Compute the mean angular velocity in the IMU frame.
        mean_ang_vel = np.zeros(3)
        for i in range(idx_begin, idx_end):
            mean_ang_vel += self.imu_msg_buffer[i].angular_velocity

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
        self.imu_msg_buffer = self.imu_msg_buffer[idx_end:]
        return cam0_R_p_c  # , cam1_R_p_c

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

    # def two_point_ransac(self, pts1, pts2, R_p_c, intrinsics,
    #         distortion_model, distortion_coeffs,
    #         inlier_error, success_probability):
    #     """
    #     Applies two point ransac algorithm to mark the inliers in the input set.

    #     Arguments:
    #         pts1: first set of points.
    #         pts2: second set of points.
    #         R_p_c: a rotation matrix takes a vector in the previous camera frame
    #             to the current camera frame.
    #         intrinsics: intrinsics of the camera.
    #         distortion_model: distortion model of the camera.
    #         distortion_coeffs: distortion coefficients.
    #         inlier_error: acceptable error to be considered as an inlier.
    #         success_probability: the required probability of success.

    #     Returns:
    #         inlier_flag: 1 for inliers and 0 for outliers.
    #     """
    #     # Check the size of input point size.
    #     assert len(pts1) == len(pts2), 'Sets of different size are used...'

    #     norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1])
    #     iter_num = int(np.ceil(np.log(1-success_probability) / np.log(1-0.7*0.7)))

    #     # Initially, mark all points as inliers.
    #     inlier_markers = [1] * len(pts1)

    #     # Undistort all the points.
    #     pts1_undistorted = self.undistort_points(pts1, intrinsics,
    #         distortion_model, distortion_coeffs)
    #     pts2_undistorted = self.undistort_points(pts2, intrinsics,
    #         distortion_model, distortion_coeffs)

    #     # Compenstate the points in the previous image with
    #     # the relative rotation.
    #     for i, pt in enumerate(pts1_undistorted):
    #         pt_h = np.array([*pt, 1.0])
    #         pt_hc = R_p_c @ pt_h
    #         pts1_undistorted[i] = pt_hc[:2]

    #     # Normalize the points to gain numerical stability.
    #     pts1_undistorted, pts2_undistorted, scaling_factor = self.rescale_points(
    #         pts1_undistorted, pts2_undistorted)

    #     # Compute the difference between previous and current points,
    #     # which will be used frequently later.
    #     pts_diff = []
    #     for pt1, pt2 in zip(pts1_undistorted, pts2_undistorted):
    #         pts_diff.append(pt1 - pt2)

    #     # Mark the point pairs with large difference directly.
    #     # BTW, the mean distance of the rest of the point pairs are computed.
    #     mean_pt_distance = 0.0
    #     raw_inlier_count = 0
    #     for i, pt_diff in enumerate(pts_diff):
    #         distance = np.linalg.norm(pt_diff)
    #         # 25 pixel distance is a pretty large tolerance for normal motion.
    #         # However, to be used with aggressive motion, this tolerance should
    #         # be increased significantly to match the usage.
    #         if distance > 50.0 * norm_pixel_unit:
    #             inlier_markers[i] = 0
    #         else:
    #             mean_pt_distance += distance
    #             raw_inlier_count += 1

    #     mean_pt_distance /= raw_inlier_count

    #     # If the current number of inliers is less than 3, just mark
    #     # all input as outliers. This case can happen with fast
    #     # rotation where very few features are tracked.
    #     if raw_inlier_count < 3:
    #         return [0] * len(inlier_markers)

    #     # Before doing 2-point RANSAC, we have to check if the motion
    #     # is degenerated, meaning that there is no translation between
    #     # the frames, in which case, the model of the RANSAC does not work.
    #     # If so, the distance between the matched points will be almost 0.
    #     if mean_pt_distance < norm_pixel_unit:
    #         for i, pt_diff in enumerate(pts_diff):
    #             if inlier_markers[i] == 0:
    #                 continue
    #             if np.linalg.norm(pt_diff) > inlier_error * norm_pixel_unit:
    #                 inlier_markers[i] = 0
    #         return inlier_markers

    #     # In the case of general motion, the RANSAC model can be applied.
    #     # The three column corresponds to tx, ty, and tz respectively.
    #     coeff_t = []
    #     for i, pt_diff in enumerate(pts_diff):
    #         coeff_t.append(np.array([
    #             pt_diff[1],
    #             -pt_diff[0],
    #             pts1_undistorted[0] * pts2_undistorted[1] -
    #             pts1_undistorted[1] * pts2_undistorted[0]]))
    #     coeff_t = np.array(coeff_t)

    #     raw_inlier_idx = np.where(inlier_markers)[0]
    #     best_inlier_set = []
    #     best_error = 1e10

    #     for i in range(iter_num):
    #         # Randomly select two point pairs.
    #         # Although this is a weird way of selecting two pairs, but it
    #         # is able to efficiently avoid selecting repetitive pairs.
    #         pair_idx1 = np.random.choice(raw_inlier_idx)
    #         idx_diff = np.random.randint(1, len(raw_inlier_idx))
    #         pair_idx2 = (pair_idx1+idx_diff) % len(raw_inlier_idx)

    #         # Construct the model.
    #         coeff_t_ = np.array([coeff_t[pair_idx1], coeff_t[pair_idx2]])
    #         coeff_tx = coeff_t_[:, 0]
    #         coeff_ty = coeff_t_[:, 1]
    #         coeff_tz = coeff_t_[:, 2]
    #         coeff_l1_norm = np.linalg.norm(coeff_t_, 1, axis=0)
    #         base_indicator = np.argmin(coeff_l1_norm)

    #         if base_indicator == 0:
    #             A = np.array([coeff_ty, coeff_tz]).T
    #             solution = np.linalg.inv(A) @ (-coeff_tx)
    #             model = [1.0, *solution]
    #         elif base_indicator == 1:
    #             A = np.array([coeff_tx, coeff_tz]).T
    #             solution = np.linalg.inv(A) @ (-coeff_ty)
    #             model = [solution[0], 1.0, solution[1]]
    #         else:
    #             A = np.array([coeff_tx, coeff_ty]).T
    #             solution = np.linalg.inv(A) @ (-coeff_tz)
    #             model = [*solution, 1.0]

    #         # Find all the inliers among point pairs.
    #         error = coeff_t @ model

    #         inlier_set = []
    #         for i, e in enumerate(error):
    #             if inlier_markers[i] == 0:
    #                 continue
    #             if np.abs(e) < inlier_error * norm_pixel_unit:
    #                 inlier_set.append(i)

    #         # If the number of inliers is small, the current model is
    #         # probably wrong.
    #         if len(inlier_set) < 0.2 * len(pts1_undistorted):
    #             continue

    #         # Refit the model using all of the possible inliers.
    #         coeff_t_ = coeff_t[inlier_set]
    #         coeff_tx_better = coeff_t_[:, 0]
    #         coeff_ty_better = coeff_t_[:, 1]
    #         coeff_tz_better = coeff_t_[:, 2]

    #         if base_indicator == 0:
    #             A = np.array([coeff_ty_better, coeff_tz_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tx_better)
    #             model_better = [1.0, *solution]
    #         elif base_indicator == 1:
    #             A = np.array([coeff_tx_better, coeff_tz_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_ty_better)
    #             model_better = [solution[0], 1.0, solution[1]]
    #         else:
    #             A = np.array([coeff_tx_better, coeff_ty_better]).T
    #             solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tz_better)
    #             model_better = [*solution, 1.0]

    #         # Compute the error and upate the best model if possible.
    #         new_error = coeff_t @ model_better
    #         this_error = np.mean([np.abs(new_error[i]) for i in inlier_set])

    #         if len(inlier_set) > best_inlier_set:
    #             best_error = this_error
    #             best_inlier_set = inlier_set

    #     # Fill in the markers.
    #     inlier_markers = [0] * len(pts1)
    #     for i in best_inlier_set:
    #         inlier_markers[i] = 1

    #     return inlier_markers

    def get_grid_size(self, img):
        """
        # Size of each grid.
        """
        grid_height = int(np.ceil(img.shape[0] / GRID_ROW))
        grid_width = int(np.ceil(img.shape[1] / GRID_COL))
        return grid_height, grid_width

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):
        """
        predictFeatureTracking Compensates the rotation between consecutive
        camera frames so that feature tracking would be more robust and fast.

        Arguments:
            input_pts: features in the previous image to be tracked.
            R_p_c: a rotation matrix takes a vector in the previous camera
                frame to the current camera frame. (matrix33)
            intrinsics: intrinsic matrix of the camera. (vec3)

        Returns:
            compensated_pts: predicted locations of the features in the
                current image based on the provided rotation.
        """
        # Return directly if there are no input features.
        if len(input_pts) == 0:
            return []

        # Intrinsic matrix.
        K = self.cam0_intrinsics
        H = K @ R_p_c @ np.linalg.inv(K)

        compensated_pts = []
        for i in range(len(input_pts)):
            p1 = np.array([*input_pts[i], 1.0])
            p2 = H @ p1
            compensated_pts.append(p2[:2] / p2[2])
        return np.array(compensated_pts, dtype=np.float32)

    # def stereo_match(self, kpts):
    #     """
    #     Matches features with stereo image pairs.
    #
    #     Arguments:
    #         kpts: points in the primary image.
    #
    #     Returns:
    #         cam1_points: points in the secondary image.
    #         inlier_markers: 1 if the match is valid, 0 otherwise.
    #     """
    #     kpts = np.array(kpts)
    #     if len(kpts) == 0:
    #         return []
    #
    #     R_cam0_cam1 = self.R_cam1_imu.T @ self.R_CAM_IMU
    #     kpts_undistorted = self.undistort_points(
    #         kpts, self.cam0_intrinsics,
    #         self.cam0_distortion_model, self.cam0_distortion_coeffs, R_cam0_cam1)
    #     cam1_points = self.distort_points(
    #         kpts_undistorted, self.cam1_intrinsics,
    #         self.cam1_distortion_model, self.cam1_distortion_coeffs)
    #     cam1_points_copy = cam1_points.copy()
    #
    #     # Track features using LK optical flow method.
    #     kpts = kpts.astype(np.float32)
    #     cam1_points = cam1_points.astype(np.float32)
    #     cam1_points, inlier_markers, _ = cv2.calcOpticalFlowPyrLK(
    #         self.curr_cam0_pyramid, self.curr_cam1_pyramid,
    #         kpts, cam1_points, **self.config.lk_params)
    #
    #     kpts_, _, _ = cv2.calcOpticalFlowPyrLK(
    #         self.curr_cam1_pyramid, self.curr_cam0_pyramid,
    #         cam1_points, kpts.copy(), **self.config.lk_params)
    #     err = np.linalg.norm(kpts - kpts_, axis=1)
    #
    #     # cam1_points_undistorted = self.undistort_points(
    #     #     cam1_points, self.cam1_intrinsics,
    #     #     self.cam1_distortion_model, self.cam1_distortion_coeffs, R_cam0_cam1)
    #     disparity = np.abs(cam1_points_copy[:, 1] - cam1_points[:, 1])
    #
    #
    #
    #     inlier_markers = np.logical_and.reduce(
    #         [inlier_markers.reshape(-1), err < 3, disparity < 20])
    #
    #     # Mark those tracked points out of the image region as untracked.
    #     img = self.cam1_curr_img_msg.image
    #     for i, point in enumerate(cam1_points):
    #         if not inlier_markers[i]:
    #             continue
    #         if (point[0] < 0 or point[0] > img.shape[1]-1 or
    #             point[1] < 0 or point[1] > img.shape[0]-1):
    #             inlier_markers[i] = 0
    #
    #     # Compute the relative rotation between the cam0 frame and cam1 frame.
    #     t_cam0_cam1 = self.R_cam1_imu.T @ (self.t_cam0_imu - self.t_cam1_imu)
    #     # Compute the essential matrix.
    #     E = skew(t_cam0_cam1) @ R_cam0_cam1
    #
    #     # Further remove outliers based on the known essential matrix.
    #     kpts_undistorted = self.undistort_points(
    #         kpts, self.cam0_intrinsics,
    #         self.cam0_distortion_model, self.cam0_distortion_coeffs)
    #     cam1_points_undistorted = self.undistort_points(
    #         cam1_points, self.cam1_intrinsics,
    #         self.cam1_distortion_model, self.cam1_distortion_coeffs)
    #
    #     norm_pixel_unit = 4.0 / (
    #         self.cam0_intrinsics[0] + self.cam0_intrinsics[1] +
    #         self.cam1_intrinsics[0] + self.cam1_intrinsics[1])
    #
    #     for i in range(len(kpts_undistorted)):
    #         if not inlier_markers[i]:
    #             continue
    #         pt0 = np.array([*kpts_undistorted[i], 1.0])
    #         pt1 = np.array([*cam1_points_undistorted[i], 1.0])
    #         epipolar_line = E @ pt0
    #         error = np.abs((pt1 * epipolar_line)[0]) / np.linalg.norm(
    #             epipolar_line[:2])
    #
    #         if error > self.config.stereo_threshold * norm_pixel_unit:
    #             inlier_markers[i] = 0
    #
    #     return cam1_points, inlier_markers
    #
    # def undistort_points(self, pts_in, intrinsics, distortion_model,
    #     distortion_coeffs, rectification_matrix=np.identity(3),
    #     new_intrinsics=np.array([1, 1, 0, 0])):
    #     """
    #     Arguments:
    #         pts_in: points to be undistorted.
    #         intrinsics: intrinsics of the camera.
    #         distortion_model: distortion model of the camera.
    #         distortion_coeffs: distortion coefficients.
    #         rectification_matrix:
    #         new_intrinsics:
    #
    #     Returns:
    #         pts_out: undistorted points.
    #     """
    #     if len(pts_in) == 0:
    #         return []
    #
    #     pts_in = np.reshape(pts_in, (-1, 1, 2))
    #     K = np.array([
    #         [intrinsics[0], 0.0, intrinsics[2]],
    #         [0.0, intrinsics[1], intrinsics[3]],
    #         [0.0, 0.0, 1.0]])
    #     K_new = np.array([
    #         [new_intrinsics[0], 0.0, new_intrinsics[2]],
    #         [0.0, new_intrinsics[1], new_intrinsics[3]],
    #         [0.0, 0.0, 1.0]])
    #
    #     if distortion_model == 'equidistant':
    #         pts_out = cv2.fisheye.undistortPoints(pts_in, K, distortion_coeffs,
    #             rectification_matrix, K_new)
    #     else:   # default: 'radtan'
    #         pts_out = cv2.undistortPoints(pts_in, K, distortion_coeffs, None,
    #             rectification_matrix, K_new)
    #     return pts_out.reshape((-1, 2))
    #
    # def distort_points(self, pts_in, intrinsics, distortion_model,
    #         distortion_coeffs):
    #     """
    #     Arguments:
    #         pts_in: points to be distorted.
    #         intrinsics: intrinsics of the camera.
    #         distortion_model: distortion model of the camera.
    #         distortion_coeffs: distortion coefficients.
    #
    #     Returns:
    #         pts_out: distorted points. (N, 2)
    #     """
    #     if len(pts_in) == 0:
    #         return []
    #
    #     K = np.array([
    #         [intrinsics[0], 0.0, intrinsics[2]],
    #         [0.0, intrinsics[1], intrinsics[3]],
    #         [0.0, 0.0, 1.0]])
    #
    #     if distortion_model == 'equidistant':
    #         pts_out = cv2.fisheye.distortPoints(pts_in, K, distortion_coeffs)
    #     else:   # default: 'radtan'
    #         homogenous_pts = cv2.convertPointsToHomogeneous(pts_in)
    #         pts_out, _ = cv2.projectPoints(homogenous_pts,
    #             np.zeros(3), np.zeros(3), K, distortion_coeffs)
    #     return pts_out.reshape((-1, 2))

    # def draw_features_stereo(self):
    #     img0 = self.curr_frame.image
    #     img1 = self.cam1_curr_img_msg.image
    #
    #     kps0 = []
    #     kps1 = []
    #     matches = []
    #     for feature in chain.from_iterable(self.curr_features):
    #         matches.append(cv2.DMatch(len(kps0), len(kps0), 0))
    #         kps0.append(cv2.KeyPoint(*feature.cam0_point, 1))
    #         kps1.append(cv2.KeyPoint(*feature.cam1_point, 1))
    #
    #     img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None, flags=2)
    #     cv2.imshow('stereo features', img)
    #     cv2.waitKey(1)


def skew(vec):
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])


def select(data, selectors):
    return [d for d, s in zip(data, selectors) if s]


if __name__ == '__main__':
    from queue import Queue
    from threading import Thread

    from config import ConfigEuRoC
    from dataset import EuRoCDataset, DataPublisher

    img_queue = Queue()
    imu_queue = Queue()

    config = ConfigEuRoC()
    image_processor = ImageProcessor(config)

    path = 'path/to/your/EuRoC_MAV_dataset/MH_01_easy'
    dataset = EuRoCDataset(path)
    dataset.set_starttime(offset=0.)

    duration = 3.
    ratio = 0.5
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)


    def process_imu(in_queue):
        while True:
            msg = in_queue.get()
            if msg is None:
                return
            print(msg.timestamp, 'imu')
            image_processor.imu_callback(msg)


    t2 = Thread(target=process_imu, args=(imu_queue,))
    t2.start()

    while True:
        msg = img_queue.get()
        if msg is None:
            break
        print(msg.timestamp, 'image')
        # cv2.imshow('left', np.hstack([x.cam0_image, x.cam1_image]))
        # cv2.waitKey(1)
        # timestamps.append(x.timestamp)
        image_processor.mono_callback(msg)

    imu_publisher.stop()
    img_publisher.stop()
    t2.join()