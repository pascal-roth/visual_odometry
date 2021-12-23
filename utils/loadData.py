# class to load the different datasets
import enum
import os
import numpy as np
import cv2
from typing import Iterator, Tuple, Optional
import dataclasses
from dateutil import parser

from utils.matrix import hom_inv

DATASET_ROOT = "./datasets"


class DatasetType(enum.Enum):
    KITTI = 0,
    MALAGA = 1,
    PARKING = 2,
    KITTI_IMU = 3


@dataclasses.dataclass(init=True, repr=True)
class FrameData:
    __slots__ = ["timestamp", "image"]
    timestamp: float
    image: np.ndarray


@dataclasses.dataclass(init=True, repr=True)
class IMUData:
    __slots__ = ["timestamp", "angular_velocity", "linear_acceleration"]
    timestamp: float
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


class Dataset:
    def __init__(self,
                 K: np.ndarray,
                 frames: Iterator[FrameData],
                 R_CAM_IMU: Optional[np.ndarray] = None,
                 T_CAM_IMU: Optional[np.ndarray] = None,
                 imu: Iterator[IMUData] = None,
                 ground_truth: np.ndarray = None):
        assert K.shape == (3, 3), "K has to be a 3x3 matrix"
        self.K = K
        self.R_CAM_IMU = R_CAM_IMU
        self.T_CAM_IMU = T_CAM_IMU
        self.frames = frames
        self.imu = imu
        self.ground_truth = ground_truth


class DatasetLoader:
    def __init__(self, dataset_type: DatasetType, path: str = None):
        if path is not None:
            self.dataset_root = path
        else:
            self.dataset_root = DATASET_ROOT
        self.dataset_type = dataset_type

    def load(self) -> Dataset:
        if self.dataset_type == DatasetType.KITTI:
            K = np.array(
                [[7.188560000000e+02, 0, 6.071928000000e+02],
                 [0, 7.188560000000e+02, 1.852157000000e+02], [0, 0, 1]],
                dtype=np.float32)
            kitti_base = os.path.abspath(
                os.path.join(self.dataset_root, "kitti"))
            return Dataset(K=K,
                           frames=DatasetLoader._load_kitti_frames(
                               path=os.path.join(kitti_base, "05", "image_0"),
                               times_path=os.path.join(kitti_base, "05",
                                                       "times.txt")),
                           ground_truth=DatasetLoader._load_kitti_ground_truth(
                               os.path.join(kitti_base, "poses", "05.txt")))
        if self.dataset_type == DatasetType.KITTI_IMU:
            K = np.array(
                [[7.188560000000e+02, 0, 6.071928000000e+02],
                 [0, 7.188560000000e+02, 1.852157000000e+02], [0, 0, 1]],
                dtype=np.float32)
            R_imu_car = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
                                  [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                                  [2.024406e-03, 1.482454e-02, 9.998881e-01]], dtype=np.float32)
            T_imu_car = np.array([[-8.086759e-01], [3.195559e-01], [-7.997231e-01]])
            R_car_cam = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05],
                                  [-2.254837e-03, -4.184312e-05, -9.999975e-01],
                                  [9.999728e-01, 7.027479e-03, -2.255075e-03]], dtype=np.float32)
            T_car_cam = np.array([[-7.137748e-03], [-7.482656e-02], [-3.336324e-01]])
            RT_IMU_CAM = np.vstack((np.hstack((R_imu_car, T_imu_car)), [0, 0, 0, 1])) @ \
                         np.vstack((np.hstack((R_car_cam, T_car_cam)), [0, 0, 0, 1]))
            RT_CAM_IMU = hom_inv(RT_IMU_CAM)
            kitti_imu_base = os.path.abspath(
                os.path.join(self.dataset_root, "kitti_IMU"))
            return Dataset(K=K,
                           R_CAM_IMU=RT_CAM_IMU[0:3, 0:3],
                           T_CAM_IMU=RT_CAM_IMU[0:3, 3],
                           frames=DatasetLoader._load_kitti_imu_frames(
                               path=os.path.join(kitti_imu_base, "image_00", "data"),
                               times_path=os.path.join(kitti_imu_base,
                                                       "image_00",
                                                       "timestamps.txt")),
                           imu=DatasetLoader._load_kitti_imu_data(
                               path=os.path.join(kitti_imu_base, "oxts",
                                                 "data"),
                               times_path=os.path.join(kitti_imu_base, "oxts",
                                                       "timestamps.txt")),
                           ground_truth=DatasetLoader._load_kitti_ground_truth(
                               os.path.join(kitti_imu_base, "poses",
                                            "05.txt")))
        elif self.dataset_type == DatasetType.MALAGA:
            K = np.array([[621.18428, 0, 404.0076], [0, 621.18428, 309.05989],
                          [0, 0, 1]],
                         dtype=np.float32)
            malaga_path = os.path.abspath(
                os.path.join(self.dataset_root,
                             "malaga-urban-dataset-extract-07"))
            return Dataset(K=K,
                           frames=DatasetLoader._load_malaga_frames(
                               malaga_path, K),
                           ground_truth=None)
        elif self.dataset_type == DatasetType.PARKING:
            parking_path = os.path.abspath(
                os.path.join(self.dataset_root, "parking"))
            K = np.array([[331.37, 0, 320], [0, 369.568, 240], [0, 0, 1]],
                         dtype=np.float32)
            return Dataset(
                K=K,
                frames=DatasetLoader._load_parking_frames(parking_path, K),
                ground_truth=DatasetLoader._load_parking_ground_truth(
                    parking_path))

    # KITTI
    @staticmethod
    def _load_kitti_ground_truth(path: str) -> np.ndarray:
        ground_truth = np.loadtxt(path)

        # load ground truth hom. transforms
        return np.array([
            np.append(np.reshape(T, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0)
            for T in ground_truth
        ],
            dtype=np.float32)

    @staticmethod
    def _load_kitti_frames(path: str, times_path: str) -> Iterator[FrameData]:
        frame_paths = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(".png")
        ])
        times = np.loadtxt(times_path)

        for i, p in enumerate(frame_paths):
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            timestamp = times[i]
            yield FrameData(timestamp, img_gray)

    # KITTI_IMU
    @staticmethod
    def _load_kitti_imu_frames(path: str,
                               times_path: str) -> Iterator[FrameData]:
        # get paths to each frame
        frame_paths = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(".png")
        ])

        # parse time string to microsecond precision
        # we require different parsing here from the above method, as
        # timestamps are in ISO format
        time_strs = []
        with open(times_path, "r") as f:
            time_strs = f.readlines()
        times = np.array(
            [parser.parse(time_str).timestamp() for time_str in time_strs])

        # build frame data iterator
        for i, p in enumerate(frame_paths):
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            timestamp = times[i]
            yield FrameData(timestamp, img_gray)

    def _load_kitti_imu_data(path: str, times_path: str) -> Iterator[IMUData]:
        # get path to each IMU sample
        imu_paths = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(".txt")
        ])

        # parse time string to microsecond precision
        time_strs = []
        with open(times_path, "r") as f:
            time_strs = f.readlines()
        times = np.array(
            [parser.parse(time_str).timestamp() for time_str in time_strs])

        # build IMU iterator
        for i, p in enumerate(imu_paths):
            imu_data = np.loadtxt(p)
            angular_velocity = imu_data[17:20]
            linear_acceleration = imu_data[11:14]
            yield IMUData(times[i], angular_velocity, linear_acceleration)

    # Malaga
    @staticmethod
    def _load_malaga_frames(
            path: str,
            K: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        frame_path = os.path.join(
            path, "malaga-urban-dataset-extract-07_rectified_800x600_Images")
        frame_paths = sorted([
            os.path.join(frame_path, f) for f in os.listdir(frame_path)
            if os.path.isfile(os.path.join(frame_path, f))
               and f.endswith("left.jpg")
        ])
        for p in frame_paths:
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield K, img_gray

    # Parking
    @staticmethod
    def _load_parking_ground_truth(path: str) -> np.ndarray:
        ground_truth_path = os.path.join(path, "poses.txt")
        ground_truth = np.loadtxt(ground_truth_path)
        # load ground truth hom. transforms
        return np.array([
            np.append(np.reshape(T, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0)
            for T in ground_truth
        ],
            dtype=np.float32)

    @staticmethod
    def _load_parking_frames(
            path: str,
            K: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        frame_path = os.path.join(path, "images")
        frame_paths = sorted([
            os.path.join(frame_path, f) for f in os.listdir(frame_path) if
            os.path.isfile(os.path.join(frame_path, f)) and f.endswith(".png")
        ])
        for p in frame_paths:
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield K, img_gray
