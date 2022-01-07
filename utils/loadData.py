# class to load the different datasets
import enum
import os
from queue import Queue
from threading import Thread
import time
import numpy as np
import cv2
from typing import Iterator, Tuple, Optional, Type, TypeVar
from dateutil import parser
from utils.message import FrameData, IMUData
import logging

from typing import List
from utils.matrix import hom_inv
from utils.transform import HomTransform

DATASET_ROOT = "./datasets"


class DatasetType(enum.Enum):
    KITTI = 0,
    MALAGA = 1,
    PARKING = 2,
    KITTI_IMU = 3


class Dataset:
    def __init__(self,
                 K: np.ndarray,
                 frames: Iterator[FrameData],
                 T_cam_body: HomTransform = None,
                 T_body_imu: HomTransform = None,
                 imu: Iterator[IMUData] = None,
                 ground_truth: np.ndarray = None):
        assert K.shape == (3, 3), "K has to be a 3x3 matrix"
        self.K = K
        self.T_cam_body = T_cam_body
        self.T_body_imu = T_body_imu
        self.T_cam_imu = T_cam_body * T_body_imu

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
            # [[7.188560000000e+02, 0, 6.071928000000e+02],
            #  [0, 7.188560000000e+02, 1.852157000000e+02], [0, 0, 1]],
            K = np.array([[9.786977e+02, 0.000000e+00, 6.900000e+02],
                          [0.000000e+00, 9.717435e+02, 2.497222e+02],
                          [0.000000e+00, 0.000000e+00, 1.000000e+00]],
                         dtype=np.float32)

            # Transformation from velodyne to camera (vehicle coords)
            T_cam_velo = HomTransform(
                R=np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
                            [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                            [2.024406e-03, 1.482454e-02, 9.998881e-01]]),
                t=np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01]))
            # Transformation from IMU to velodyne
            T_velo_imu = HomTransform(
                R=np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05],
                            [-2.254837e-03, -4.184312e-05, -9.999975e-01],
                            [9.999728e-01, 7.027479e-03, -2.255075e-03]]),
                t=np.array([-7.137748e-03, -7.482656e-02, -3.336324e-01]))

            kitti_imu_base = os.path.abspath(
                os.path.join(self.dataset_root, "kitti_IMU"))

            # get paths to each frame
            kitti_frame_path = os.path.join(kitti_imu_base, "image_00", "data")
            frame_paths = sorted([
                os.path.join(kitti_frame_path, f)
                for f in os.listdir(kitti_frame_path)
                if os.path.isfile(os.path.join(kitti_frame_path, f))
                and f.endswith(".png")
            ])

            # parse time string to microsecond precision
            # we require different parsing here from the above method, as
            # timestamps are in ISO format
            kitti_times_path = os.path.join(kitti_imu_base, "image_00",
                                            "timestamps.txt")
            with open(kitti_times_path, "r") as f:
                frame_timestamps = np.array([
                    parser.parse(time_str).timestamp()
                    for time_str in f.readlines()
                ])

            # get path to each IMU sample
            kitti_imu_path = os.path.join(kitti_imu_base, "oxts", "data")
            imu_paths = sorted(
                (os.path.join(kitti_imu_path, f)
                 for f in os.listdir(kitti_imu_path)
                 if os.path.isfile(os.path.join(kitti_imu_path, f))
                 and f.endswith(".txt")))

            # parse time string to microsecond precision
            kitti_imu_times_path = os.path.join(kitti_imu_base, "oxts",
                                                "timestamps.txt")
            with open(kitti_imu_times_path, "r") as f:
                imu_times = np.array([
                    parser.parse(time_str).timestamp()
                    for time_str in f.readlines()
                ])

            return Dataset(
                K=K,
                T_body_imu=T_velo_imu,
                T_cam_body=T_cam_velo,
                frames=DatasetLoader._load_kitti_imu_frames(
                    frame_paths=frame_paths, timestamps=frame_timestamps),
                imu=DatasetLoader._load_kitti_imu_data(imu_paths=imu_paths,
                                                       timestamps=imu_times),
                ground_truth=DatasetLoader._load_kitti_ground_truth(
                    os.path.join(kitti_imu_base, "poses", "05.txt")))
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
    def _load_kitti_imu_frames(frame_paths: List[str],
                               timestamps: List[float]) -> Iterator[FrameData]:
        # only load actual data on each frame to synchronize with other data streams
        for i, p in enumerate(frame_paths):
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            timestamp = timestamps[i]
            yield FrameData(timestamp, img_gray)

    @staticmethod
    def _load_kitti_imu_data(imu_paths: List[str],
                             timestamps: List[float]) -> Iterator[IMUData]:
        # only load actual data on each frame to synchronize with other data streams
        for i, p in enumerate(imu_paths):
            imu_data = np.loadtxt(p)
            angular_velocity = imu_data[17:20]
            linear_acceleration = imu_data[11:14]
            yield IMUData(timestamps[i], angular_velocity, linear_acceleration)

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


# Generic type var for any data type
T = TypeVar("T", FrameData, IMUData)


class DataPublisher:
    def __init__(self,
                 data: Iterator[T],
                 out_queue: 'Queue[T]',
                 type_name: str = "") -> None:
        """Publishes data from the given iterator at the given timestamps
        to the queue out_queue

        Args:
            data (Iterator[T]): Iterator to get data from
            out_queue (Queue[T]): Queue to write data into 
        """
        self.data = data
        self.out_queue = out_queue
        self.type_name = type_name
        self.running = False
        self.thread = Thread(target=self._publish)
        self.prev_time = None

    def start(self):
        """Start publishing to out_queue
        """
        self.running = True
        self.thread.start()

    def stop(self):
        """Stop publishing to out_queue
        """
        if self.running:
            self.thread.join()
            self.running = False
        self.out_queue.put(None)

    def _publish(self):
        while self.running:
            # wait if the queue could not be consumed fast enough
            if self.out_queue.full():
                time.sleep(1e-1)
                continue

            try:
                data = next(self.data)
            except StopIteration:
                self.out_queue.put(None)
                logging.warning(f"Stopped {self.type_name} DataPublisher")
                return

            # sleep until the data is supposed to be sent
            if self.prev_time is not None:
                interval = data.timestamp - self.prev_time
                time.sleep(interval)

            # write data into queue
            self.out_queue.put(data)

            self.prev_time = data.timestamp
