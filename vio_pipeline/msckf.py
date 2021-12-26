import numpy as np
from typing import Dict, List

from numpy.lib.function_base import cov
from params import *
from scipy.stats import chi2
from utils.quaternion import Quaternion
from utils.message import IMUData, FeatureData
from utils.matrix import skew

import time

from utils.transform import HomTransform


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

    def __init__(self, id=None):
        self.id = id
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

        # Orientation form world frame to camera frame
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
        state_cov = np.zeros_like(self.state_cov)
        state_cov[3:6, 3:6] = GYRO_BIAS_COV * np.identity(3)
        state_cov[6:9, 6:9] = VELOCITY_COV * np.identity(3)
        state_cov[9:12, 9:12] = ACC_BIAS_COV * np.identity(3)
        state_cov[15:18, 15:18] = EXTRINSIC_ROTATION_COV * np.identity(3)
        state_cov[18:21, 18:21] = EXTRINSIC_TRANSLATION_COV * np.identity(3)
        self.state_cov = state_cov

    def reset_noise_cov(self) -> None:
        """
        Resets noise covariance to values in params.
        """
        continuous_noise_cov = np.eye(*self.continuous_noise_cov.shape)
        continuous_noise_cov[:3, :3] *= GYRO_NOISE
        continuous_noise_cov[3:6, 3:6] *= GYRO_BIAS_NOISE
        continuous_noise_cov[6:9, 6:9] *= ACC_NOISE
        continuous_noise_cov[9:, 9:] *= ACC_BIAS_NOISE
        self.continuous_noise_cov = continuous_noise_cov


class MSCKF:
    def __init__(self, R_CAM_IMU):

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer: List[IMUData] = []

        # Features used
        self.map_server = dict()  # featureid, feature

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

    def imu_callback(self, imu_msg: IMUData):
        # IMU messages are only processed as soon as a new image
        # becomes available.
        self.imu_msg_buffer.append(imu_msg)

        # initialize gravity estimate
        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self._init_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg: FeatureData):
        if not self.is_gravity_set:
            return
        start_time = time.time()

        # Start the system if the first image has been received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        self.batch_imu_processing(feature_msg.timestamp)

        self.state_augmentation(feature_msg.timestamp)

        self.add_feature_observations(feature_msg)

        self.remove_lost_features()

        self.prune_cam_state_buffer()

        try:
            # Publish the odometry data
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary
            self.online_reset()

    def _init_gravity_and_bias(self):
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

    def batch_imu_processing(self, time_bound: time.time):
        """Propagate the EKF state

        Args:
            time_bound (time.time): [description]
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
                      input_acc: np.ndarray):
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

    def predict_new_state(self, dt: float, gyro: np.ndarray, acc: np.ndarray):
        """Propagate the state using 4th order Runge-Kutta

        Args:
            dt (float): [description]
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

    def state_augmentation(self, time: float):
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

        # Resize the state covariance matrix.
        old_size = self.state_server.state_cov.shape[0]
        state_cov = np.zeros((old_size + 6, old_size + 6))
        state_cov[:old_size, :old_size] = self.state_server.state_cov

        # Fill in augmented state covariance
        state_cov[old_size:, :old_size] = J @ state_cov[:21, :old_size]
        state_cov[:old_size, old_size:] = state_cov[old_size:, :old_size].T
        state_cov[old_size:, old_size:] = J @ state_cov[:21, :21] @ J.T

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.

    def add_feature_observations(self, feature_msg: FeatureData):
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0
        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                # TODO add features here

                pass 
        pass


    def remove_lost_features(self):
        pass

    def prune_cam_state_buffer(self):
        pass