from time import time

import numpy as np

from utils.quaternion import Quaternion
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
    T_imu_to_body = HomTransform.identity()

    def __init__(self, state_id: int = None):
        self.id = state_id
        # time when state was recorded
        self.timestamp: time.time = None

        # Orientation from world frame to IMU frame
        self.orientation_world_to_imu = Quaternion.identity()
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

        # # Transformation from IMU to camera in IMU frame
        self.R_imu_to_cam = np.eye(3)
        self.t_imu_to_cam = np.zeros(3)


class CameraState:
    """
    Stores the camera state at the given timestamp
    """
    def __init__(self, id: int, timestamp: float, orientation: Quaternion,
                 position: np.ndarray, orientation_null: Quaternion,
                 position_null: np.ndarray) -> None:
        self.id = id
        self.timestamp: float = timestamp

        # Orientation from world frame to camera frame
        self.orientation_world_to_cam = orientation
        # Position of the camera frame in world frame
        self.position = position

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = orientation_null
        self.position_null = position_null
