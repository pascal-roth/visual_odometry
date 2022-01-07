import numpy as np
from copy import copy
from typing import Dict, List, Tuple
import warnings
import pprint

from numpy.lib.function_base import cov
from params import *
from scipy.stats import chi2
from utils.quaternion import Quaternion
from utils.message import IMUData, FeatureData, LandmarkData, PoseData
from utils.matrix import skew

import time

from utils.transform import HomTransform
from vio_pipeline.feature import Feature


class IMUState:
    """
    Stores the state estimate of the IMU at a time step. 
    """
    # id for the next IMUState
    next_id = 0
    # gravity in world frame (class level)
    gravity = np.array([0, 0, -9.81])
    # Transformation from IMU frame to body frame
    # z-axis of the body frame should point upwards
    T_body_imu = HomTransform.identity()

    def __init__(self, state_id: int = None):
        self.id = state_id
        # time when state was recorded
        self.timestamp: time.time = None

        # Orientation from world frame to IMU frame
        self.orientation = Quaternion.identity()
        # position of the IMU frame in world frame
        self.position = np.zeros(3)
        # velocity of the IMU in world frame
        self.velocity = np.zeros(3)

        # Biases for angular velocity
        self.gyro_bias = np.zeros(3)
        # Biases for acceleration
        self.acc_bias = np.zeros(3)

        # Modifyers for the observability matrix to have a proper
        # null-space
        self.orientation_null = Quaternion.identity()
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # # Transformation between the IMU and the camera
        self.R_cam_imu = np.eye(3)
        self.t_imu_cam = np.zeros(3)


class CameraState:
    def __init__(self, id: int, timestamp: float, orientation: Quaternion,
                 position: np.ndarray, orientation_null: Quaternion,
                 position_null: np.ndarray) -> None:
        self.id = id
        self.timestamp: float = timestamp

        # Orientation from world frame to camera frame
        self.orientation = orientation
        # Position of the camera frame in world frame
        self.position = position

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = orientation_null
        self.position_null = position_null


class StateServer:
    def __init__(self, imu_state: IMUState = None) -> None:
        self.imu_state = imu_state if imu_state is not None else IMUState()
        self.cam_states: Dict[int, CameraState] = dict()

        self.state_cov = np.zeros((21, 21))
        self.continuous_noise_cov = np.zeros((12, 12))

    def reset_state_cov(self) -> None:
        """
        Reset state covariance to values in params.
        """
        state_cov = np.zeros((21, 21))
        # state_cov[0:3, 0:3] = error quaternion
        state_cov[3:6, 3:6] = GYRO_BIAS_COV * np.identity(3)
        state_cov[6:9, 6:9] = VELOCITY_COV * np.identity(3)
        state_cov[9:12, 9:12] = ACC_BIAS_COV * np.identity(3)
        # state_cov[12:15, 12:15] = pos
        state_cov[15:18, 15:18] = EXTRINSIC_ROTATION_COV * np.identity(3)
        state_cov[18:21, 18:21] = EXTRINSIC_TRANSLATION_COV * np.identity(3)
        self.state_cov = state_cov

    def reset_noise_cov(self) -> None:
        """
        Resets noise covariance to values in params.
        """
        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= GYRO_NOISE
        continuous_noise_cov[3:6, 3:6] *= GYRO_BIAS_NOISE
        continuous_noise_cov[6:9, 6:9] *= ACC_NOISE
        continuous_noise_cov[9:, 9:] *= ACC_BIAS_NOISE
        self.continuous_noise_cov = continuous_noise_cov


class MSCKF:
    def __init__(self, R_CAM_IMU) -> None:
        self.optimization_config = OptimizationParams()

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer: List[IMUData] = []

        # Features used
        self.map_server: Dict[int, Feature] = dict()  # featureid, feature

        # Chi squared test table
        self.chi_squared_test_table: Dict[float] = {
            i: chi2.ppf(1 - CHI2_CONFIDENCE, i)
            for i in range(100)
        }

        # Set initial IMU state
        imu_state = IMUState()
        # TODO add transform from IMU to camera  # INFO: not sure if thats the correct transformation
        imu_state.R_cam_imu = R_CAM_IMU
        # TODO add transform from IMU to body frame
        self.state_server = StateServer(imu_state=imu_state)
        self.state_server.reset_state_cov()
        self.state_server.reset_noise_cov()

        # Tracking rate
        self.tracking_rate = None
        # Indicate if the gravity vector has been set
        self.is_gravity_set = False
        # Indicate if the image received is the first one
        self.is_first_img = True

    def imu_callback(self, imu_msg: IMUData) -> None:
        # IMU messages are only processed as soon as a new image
        # becomes available.
        self.imu_msg_buffer.append(imu_msg)

        # initialize gravity estimate
        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 10:
                self._init_gravity_and_bias()
                self.is_gravity_set = True

    def _init_gravity_and_bias(self) -> None:
        """
        Initialize IMU biases and initial orientation based on
        the first few IMU samples.
        """

        sum_angular_vel = np.sum(
            [msg.angular_velocity for msg in self.imu_msg_buffer], axis=0)
        sum_linear_acc = np.sum(
            [msg.linear_acceleration for msg in self.imu_msg_buffer], axis=0)

        # gyro bias is mean angular velocity
        gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = gyro_bias

        # gravity in IMU frame is mean linear velocity
        gravity_imu = sum_linear_acc / len(self.imu_msg_buffer)

        # Initialize the initial orientation such that the estimation
        # is consistent with the inertial frame
        gravity_norm = np.linalg.norm(gravity_imu)
        IMUState.gravity = np.array([0, 0, -gravity_norm])
        self.state_server.imu_state.orientation = Quaternion.from_two_vectors(
            -IMUState.gravity, gravity_imu)
        print(f"Initialized gravity vector: {IMUState.gravity},")
        print(
            f"initial orientation: {self.state_server.imu_state.orientation}")

    def feature_callback(
            self, feature_msg: FeatureData) -> Tuple[PoseData, LandmarkData]:
        if not self.is_gravity_set:
            return None, None
        start_time = time.time()

        # Start the system if the first image has been received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.start_time = feature_msg.timestamp
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        try:
            t = time.time()

            # Propogate the IMU state.
            # that are received before the image msg.
            self.batch_imu_processing(feature_msg.timestamp)
            # print(f"---batch_imu_processing    {time.time() - t:.2f}s")
            # t = time.time()

            # Augment the state vector.
            self.state_augmentation(feature_msg.timestamp)
            # print(f"---state_augmentation      {time.time() - t:.2f}s")
            # t = time.time()

            # Add new observations for existing features or new features
            # in the map server.
            self.add_feature_observations(feature_msg)
            # print(f"---add_feature_observations{time.time() - t:.2f}s")
            # t = time.time()

            # Perform measurement update if necessary.
            # And prune features and camera states.
            self.remove_lost_features()
            # print(f"---remove_lost_features    {time.time() - t:.2f}s")

            t = time.time()
            self.prune_cam_state_buffer()
            # print(f"---prune_cam_state_buffer  {time.time() - t:.2f}s")
            # print(
            #     f"---msckf elapsed:          {time.time() - start_time:.2f}s, delta_t: {feature_msg.timestamp - self.start_time:.2f}s"
            # )
            # print()

            # Publish the odometry data
            return self.publish_pose(
                feature_msg.timestamp), self.publish_landmarks(
                    feature_msg.timestamp)
        except RuntimeError as e:
            print(e)
            self.online_reset(force=True)
        finally:
            # Reset the system if necessary
            self.online_reset()

    def batch_imu_processing(self, time_bound: float) -> None:
        """Propagate the EKF state

        Args:
            time_bound (float): [description]
        """

        consumed_msgs = 0
        for msg in self.imu_msg_buffer:
            imu_time = msg.timestamp
            if imu_time < self.state_server.imu_state.timestamp:
                consumed_msgs += 1
                continue
            if imu_time > time_bound:
                break

            # Execute process model
            self.process_model(imu_time, msg.angular_velocity,
                               msg.linear_acceleration)
            consumed_msgs += 1

            # Propagate state info
            self.state_server.imu_state.timestamp = imu_time

        self.state_server.imu_state.id = IMUState.next_id
        IMUState.next_id += 1

        # remove consumed messages
        self.imu_msg_buffer = self.imu_msg_buffer[consumed_msgs:]

    def process_model(self, time: float, input_gyro: np.ndarray,
                      input_acc: np.ndarray) -> None:
        imu_state = self.state_server.imu_state
        dt = time - imu_state.timestamp

        gyro = input_gyro - imu_state.gyro_bias
        acc = input_acc - imu_state.acc_bias

        # Compute discrete transition and noise covariance matrix
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))

        R_imu_world = imu_state.orientation.to_rotation()

        F[:3, :3] = -skew(gyro)
        F[:3, 3:6] = -np.identity(3)
        F[6:9, :3] = -R_imu_world.T @ skew(acc)
        F[6:9, 9:12] = -R_imu_world.T
        F[12:15, 6:9] = np.identity(3)

        G[:3, :3] = -np.identity(3)
        G[3:6, 3:6] = np.identity(3)
        G[6:9, 6:9] = -R_imu_world.T
        G[9:12, 9:12] = np.identity(3)

        # Approximate matrix exponentail using 3rd order
        Fdt = F * dt
        Fdt_square = Fdt @ Fdt
        Fdt_cube = Fdt_square @ Fdt
        Phi = np.eye(21) + Fdt + Fdt_square / 2. + Fdt_cube / 6.

        self.predict_new_state(dt, gyro, acc)

        # Modify the transition matrix
        R_kk_1 = imu_state.orientation_null.to_rotation()
        Phi[:3, :3] = imu_state.orientation.to_rotation() @ R_kk_1

        u = R_kk_1 @ IMUState.gravity
        s = u / (u @ u)

        A1 = Phi[6:9, :3]
        w1 = skew(imu_state.velocity_null -
                  imu_state.velocity) @ IMUState.gravity
        Phi[6:9, :3] = A1 - (A1 @ u - w1)[:, None] * s

        A2 = Phi[12:15, :3]
        w2 = skew(dt * imu_state.velocity_null + imu_state.position_null -
                  imu_state.position) @ IMUState.gravity
        Phi[12:15, :3] = A2 - (A2 @ u - w2)[:, None] * s

        # Propagate the state covariance matrix
        Q = Phi @ G @ self.state_server.continuous_noise_cov @ G.T @ Phi * dt
        self.state_server.state_cov[:21, :21] = (
            Phi @ self.state_server.state_cov[:21, :21] @ Phi.T + Q)

        if len(self.state_server.cam_states) > 0:
            self.state_server.state_cov[:21, 21:] = (
                Phi @ self.state_server.state_cov[:21, 21:])
            self.state_server.state_cov[21:, :21] = (
                self.state_server.state_cov[21:, :21] @ Phi.T)

        # Fix the covariance to be symmetric (take sum orthogonal and mean)
        self.state_server.state_cov = (self.state_server.state_cov +
                                       self.state_server.state_cov.T) / 2.

        # Update the state correspondences to null space.
        self.state_server.imu_state.orientation_null = imu_state.orientation
        self.state_server.imu_state.position_null = imu_state.position
        self.state_server.imu_state.velocity_null = imu_state.velocity

    def predict_new_state(self, dt: float, gyro: np.ndarray,
                          acc: np.ndarray) -> None:
        """Propagate the state using 4th order Runge-Kutta

        Args:
            dt (float): [descri        # print('+++publish:')
        # print('   timestamp:', imu_state.timestamp)
        # print('   orientation:', imu_state.orientation)
        # print('   position:', imu_state.position)
        # print('   velocity:', imu_state.velocity)
        # print()ption]
            gyro (np.ndarray): [description]
            acc (np.ndarray): [description]
        """
        # TODO does scipy have 4th order RK implementation
        # TODO: Will performing the forward integration using
        # the inverse of the quaternion give better accuracy?
        gyro_norm = np.linalg.norm(gyro)
        Omega = np.zeros((4, 4))
        Omega[:3, :3] = -skew(gyro)
        Omega[:3, 3] = gyro
        Omega[3, :3] = -gyro

        q = self.state_server.imu_state.orientation
        v = self.state_server.imu_state.velocity
        p = self.state_server.imu_state.position

        if gyro_norm > 1e-5:
            dq_dt = Quaternion(
                (np.cos(gyro_norm * dt * 0.5) * np.identity(4) +
                 np.sin(gyro_norm * dt * 0.5) / gyro_norm * Omega) @ q.q)
            dq_dt2 = Quaternion(
                (np.cos(gyro_norm * dt * 0.25) * np.identity(4) +
                 np.sin(gyro_norm * dt * 0.25) / gyro_norm * Omega) @ q.q)
        else:
            dq_dt = Quaternion(
                np.cos(gyro_norm * dt * 0.5) *
                (np.identity(4) + Omega * dt * 0.5) @ q.q)
            dq_dt2 = Quaternion(
                np.cos(gyro_norm * dt * 0.25) *
                (np.identity(4) + Omega * dt * 0.25) @ q.q)

        dR_dt_transpose = dq_dt.to_rotation().T
        dR_dt2_transpose = dq_dt2.to_rotation().T

        # k1 = f(tn, yn)
        k1_p_dot = v
        k1_v_dot = q.to_rotation().T @ acc + IMUState.gravity

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k1_v = v + k1_v_dot * dt / 2.
        k2_p_dot = k1_v
        k2_v_dot = dR_dt2_transpose @ acc + IMUState.gravity

        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k2_v = v + k2_v_dot * dt / 2
        k3_p_dot = k2_v
        k3_v_dot = dR_dt2_transpose @ acc + IMUState.gravity

        # k4 = f(tn+dt, yn+k3*dt)
        k3_v = v + k3_v_dot * dt
        k4_p_dot = k3_v
        k4_v_dot = dR_dt_transpose @ acc + IMUState.gravity

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q = dq_dt.normalized()
        v = v + (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot) * dt / 6.
        p = p + (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot) * dt / 6.

        self.state_server.imu_state.orientation = q
        self.state_server.imu_state.velocity = v
        self.state_server.imu_state.position = p

    def state_augmentation(self, time: float) -> None:
        """Measurement update, state augmentation

        Args:
            time (float): [description]
        """
        imu_state = self.state_server.imu_state
        R_cam_imu = imu_state.R_cam_imu
        t_imu_cam = imu_state.t_imu_cam

        # Add a new state to the state server
        R_imu_world = imu_state.orientation.to_rotation()
        R_cam_world = R_cam_imu @ R_imu_world
        t_world_cam = imu_state.position + R_imu_world.T @ t_imu_cam

        cam_state = CameraState(
            id=imu_state.id,
            timestamp=time,
            orientation=Quaternion.from_rotation(R_cam_world),
            position=t_world_cam,
            orientation_null=Quaternion.from_rotation(R_cam_world),
            position_null=t_world_cam)
        self.state_server.cam_states[imu_state.id] = cam_state

        # Update covariance matrix of the state
        # To simplify computation, the matrix J below is the nontrivial block
        # in Equation (16) of "MSCKF" paper.
        J = np.zeros((6, 21))
        J[:3, :3] = R_cam_world
        J[:3, 15:18] = np.identity(3)
        J[3:6, :3] = skew(R_imu_world.T @ t_imu_cam)
        J[3:6, 12:15] = np.identity(3)
        J[3:6, 18:21] = np.identity(3)

        # TODO: is this correct?
        # Resize the state covariance matrix.
        old_size, _ = self.state_server.state_cov.shape
        state_cov = np.zeros((old_size + 6, old_size + 6))
        state_cov[:old_size, :old_size] = self.state_server.state_cov

        # Fill in augmented state covariance
        state_cov[old_size:, :old_size] = J @ state_cov[:21, :old_size]
        state_cov[:old_size, old_size:] = state_cov[old_size:, :old_size].T
        state_cov[old_size:, old_size:] = J @ state_cov[:21, :21] @ J.T

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.

    def add_feature_observations(self, feature_msg: FeatureData) -> None:
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0
        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                map_feature = Feature(feature.id, self.optimization_config)
                map_feature.observations[state_id] = feature.as_array()
                self.map_server[feature.id] = map_feature
            else:
                # This is a new feature
                self.map_server[
                    feature.id].observations[state_id] = feature.as_array()
                tracked_feature_num += 1

        # update tracking rate, when curr_feature_num can also be none
        try:
            self.tracking_rate = tracked_feature_num / curr_feature_num
        except ZeroDivisionError:
            self.tracking_rate = 1. if tracked_feature_num > 0 else 0.

    def measurement_jacobian(
            self, cam_state_id: int,
            feature_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = cam_state.orientation.to_rotation()
        t_c0_w = cam_state.position

        # Cam1 pose.
        # R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        # t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        # p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((2, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        # dz_dpc1 = np.zeros((4, 3))
        # dz_dpc1[2, 0] = 1 / p_c1[2]
        # dz_dpc1[3, 1] = 1 / p_c1[2]
        # dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        # dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        # dpc1_dxc = np.zeros((3, 6))
        # dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        # dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        # dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc  #+ dz_dpc1 @ dpc1_dxc  # shape: (2, 6)
        H_f = dz_dpc0 @ dpc0_dpg  #+ dz_dpc1 @ dpc1_dpg  # shape: (2, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x  # shape: (2, 6)
        u = np.zeros(6)
        u[:3] = cam_state.orientation_null.to_rotation() @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:2, 3:6]

        # Compute the residual.
        r = z - (p_c0[:2] / p_c0[2])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(
            self, feature_id: int,
            cam_state_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id
        # camera has actually seen this feature.
        valid_cam_state_ids = [
            cam_id for cam_id in cam_state_ids
            if cam_id in feature.observations
        ]

        jacobian_row_size = 2 * len(valid_cam_state_ids)

        # TODO: not sure if right values here, in mono cpp implementation they differ
        H_xj = np.zeros(
            (jacobian_row_size, 21 + len(self.state_server.cam_states) * 6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count + 2,
                 21 + 6 * idx:21 + 6 * (idx + 1)] = H_xi
            H_fj[stack_count:stack_count + 2, :3] = H_fi
            r_j[stack_count:stack_count + 2] = r_i
            stack_count += 2

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        try:
            U, _, _ = np.linalg.svd(H_fj)
        except np.linalg.LinAlgError:
            raise ValueError

        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H: np.ndarray, r: np.ndarray) -> None:
        if H.size == 0 or r.size == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity as in Equation (28), (29).
        if H.shape[0] > H.shape[1]:
            # QR decomposition
            Q, R = np.linalg.qr(
                H, mode='reduced')  # if M > N, return (M, N), (N, N)
            H_thin = R  # shape (N, N)
            r_thin = Q.T @ r  # shape (N,)
        else:
            H_thin = H  # shape (M, N)
            r_thin = r  # shape (M)

        # Compute the Kalman gain.
        P = self.state_server.state_cov
        S = H_thin @ P @ H_thin.T + (OBSERVATION_NOISE *
                                     np.identity(len(H_thin)))
        K_transpose = np.linalg.solve(S, H_thin @ P)
        K = K_transpose.T  # shape (N, K)

        # Compute the error of the state.
        delta_x = K @ r_thin

        # Update the IMU state.
        delta_x_imu = delta_x[:21]
        delta_vel = delta_x_imu[6:9]
        delta_pos = delta_x_imu[12:15]
        norm_delta_vel = np.linalg.norm(delta_vel)
        norm_delta_pos = np.linalg.norm(delta_pos)
        if norm_delta_vel > VELOCITY_DELTA_THRESHOLD or norm_delta_pos > POSITION_DELTA_THRESHOLD:
            warnings.warn(
                f"Update change is too large: ||delta_vel|| = {norm_delta_vel}, "
                f"||delta_pos|| = {norm_delta_pos}, "
                f"small angle quaternion approximation will be inaccurate")

        dq_imu = Quaternion.small_angle_quaternion(delta_x_imu[:3])
        imu_state = self.state_server.imu_state
        imu_state.orientation = dq_imu * imu_state.orientation
        imu_state.gyro_bias += delta_x_imu[3:6]
        imu_state.velocity += delta_x_imu[6:9]
        imu_state.acc_bias += delta_x_imu[9:12]
        imu_state.position += delta_x_imu[12:15]

        dq_extrinsic = Quaternion.small_angle_quaternion(delta_x_imu[15:18])
        imu_state.R_cam_imu = dq_extrinsic.to_rotation() @ imu_state.R_cam_imu
        imu_state.t_imu_cam += delta_x_imu[18:21]

        # Update the camera states.
        for i, (cam_id,
                cam_state) in enumerate(self.state_server.cam_states.items()):
            delta_x_cam = delta_x[21 + i * 6:27 + i * 6]
            dq_cam = Quaternion.small_angle_quaternion(delta_x_cam[:3])
            cam_state.orientation = dq_cam * cam_state.orientation
            cam_state.position += delta_x_cam[3:]

        # Update state covariance.
        I_KH = np.identity(len(K)) - K @ H_thin
        state_cov = I_KH @ self.state_server.state_cov @ I_KH.T + (
            K @ K.T * OBSERVATION_NOISE)
        # state_cov = I_KH @ self.state_server.state_cov  # ?

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.
        print(
            f"Updated IMU_state, delta position: {norm_delta_pos}, delta velocity:{norm_delta_vel}"
        )

    def gating_test(self, H: np.ndarray, r: np.ndarray, dof: int) -> bool:
        # try:
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = OBSERVATION_NOISE * np.identity(H.shape[0])
        gamma = r @ np.linalg.solve(P1 + P2, r)
        # gamma = np.abs(gamma)
        threshold = self.chi_squared_test_table[dof]
        return gamma < threshold
        # except ValueError:
        #     return False

    def remove_lost_features(self) -> None:
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_baseline(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available
                # measurements.
                valid_pos = feature.initialize_position(
                    self.state_server.cam_states)
                if not valid_pos:
                    invalid_feature_ids.append(feature.id)
                    continue
            jacobian_row_size += 4 * len(feature.observations) - 3
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros(
            (jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            # if feature_id == 2461:
            #     print('STOP')

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids) - 1):
                H_x[stack_count:stack_count +
                    H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the execution time.
            if stack_count > 1500:
                warnings.warn("Stack count too large, stopping")
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        print(
            f"num lost features: {len(processed_feature_ids)}, num invalid features: {len(invalid_feature_ids)}, stack count: {stack_count}"
        )

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self) -> List[int]:
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = cam_state_pairs[key_cam_state_idx][
            1].orientation.to_rotation()

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = cam_state_pairs[cam_state_idx][
                1].orientation.to_rotation()

            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(
                Quaternion.from_rotation(rotation @ key_rotation.T).q[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(
                    cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        return sorted(rm_cam_state_ids)

    def prune_cam_state_buffer(self) -> None:
        if len(self.state_server.cam_states) < MAX_CAM_STATE_SIZE:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialized
                if not feature.check_baseline(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if not ret:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4 * len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros(
            (jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id,
                                              involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count +
                    H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6 * idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset(self) -> None:
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        self.state_server.imu_state = copy(self.state_server.imu_state)

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.state_server.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self, force=False) -> None:
        """
        Reset the system online if the uncertainty is too large.
        """
        if not force:
            # Never perform online reset if position std threshold is non-positive.
            if POSITION_STD_THRESHOLD <= 0:
                return

            try:
                # Check the uncertainty of positions to determine if
                # the system can be reset.
                position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
                position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
                position_z_std = np.sqrt(self.state_server.state_cov[14, 14])
                max_pos_std = max(position_x_std, position_y_std,
                                  position_z_std)
                state_cov_std = np.sqrt(self.state_server.state_cov.diagonal())
                uncertainties = {
                    "dq": state_cov_std[0:3],
                    "gyro bias": state_cov_std[3:6],
                    "velocity": state_cov_std[6:9],
                    "acc bias": state_cov_std[9:12],
                    "pos": state_cov_std[12:15],
                    "extrinsic rot": state_cov_std[15:18],
                    "extrinsic trans": state_cov_std[18:21]
                }
                uncertainties = {
                    k: np.max(v)
                    for k, v in uncertainties.items()
                }
                # print(f"State uncertainties:")
                # pprint.pprint(uncertainties)
                # print(f"Tracked features: {len(self.map_server.values())}")
                if max_pos_std < POSITION_STD_THRESHOLD:
                    return
            except RuntimeWarning:
                pass

        else:
            print('Forced online reset')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all existing features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.state_server.reset_state_cov()

    def publish_pose(self, time: float) -> PoseData:
        imu_state = self.state_server.imu_state
        # print('+++publish:')
        # print('   timestamp:', imu_state.timestamp)
        # print('   orientation:', imu_state.orientation)
        # print('   position:', imu_state.position)
        # print('   velocity:', imu_state.velocity)
        # print()

        T_i_w = HomTransform(imu_state.orientation.to_rotation().T,
                             imu_state.position)
        T_b_w = IMUState.T_body_imu * T_i_w * IMUState.T_body_imu.inverse()
        body_velocity = IMUState.T_body_imu.R @ imu_state.velocity

        R_w_c = imu_state.R_cam_imu @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_imu_cam
        T_c_w = HomTransform(R_w_c.T, t_c_w)

        return PoseData(time, T_b_w, body_velocity, T_c_w)

    def publish_landmarks(self, time: float) -> LandmarkData:
        landmarks = [
            feature.position for feature in self.map_server.values()
            if feature.is_initialized
        ]
        return LandmarkData(time, np.asarray(landmarks))
