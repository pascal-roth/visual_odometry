# class to load the different datasets
import enum
import os
import numpy as np
import cv2
from typing import Iterator, Tuple, Optional

DATASET_ROOT = "./datasets"


class DatasetType(enum.Enum):
    KITTI = 0,
    MALAGA = 1,
    PARKING = 2


class Dataset:
    def __init__(self, K: np.ndarray, frames: Iterator[Tuple[np.ndarray, np.ndarray]], T: Optional[np.ndarray]):
        assert K.shape == (3, 3), "K has to be a 3x3 matrix"
        self.K = K
        self.frames = frames
        self.T = T


class DatasetLoader:
    def __init__(self, dataset_type: DatasetType, path: str = None):
        if path is not None:
            self.dataset_root = path
        else:
            self.dataset_root = DATASET_ROOT
        self.dataset_type = dataset_type

    def load(self) -> Dataset:
        if self.dataset_type == DatasetType.KITTI:
            K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                          [0, 7.188560000000e+02, 1.852157000000e+02],
                          [0, 0, 1]])
            kitti_path = os.path.abspath(os.path.join(self.dataset_root, "kitti"))
            return Dataset(K=K,
                           frames=DatasetLoader._load_kitti_frames(kitti_path, K),
                           T=DatasetLoader._load_kitti_ground_truth(kitti_path))
        elif self.dataset_type == DatasetType.MALAGA:
            K = np.array([[621.18428, 0, 404.0076],
                          [0, 621.18428, 309.05989],
                          [0, 0, 1]])
            malaga_path = os.path.abspath(os.path.join(self.dataset_root, "malaga-urban-dataset-extract-07"))
            return Dataset(K=K,
                           frames=DatasetLoader._load_malaga_frames(malaga_path, K),
                           T=None)
        elif self.dataset_type == DatasetType.PARKING:
            parking_path = os.path.abspath(os.path.join(self.dataset_root, "parking"))
            K = np.array([[331.37, 0, 320],
                          [0, 369.568, 240],
                          [0, 0, 1]])
            return Dataset(K=K,
                           frames=DatasetLoader._load_parking_frames(parking_path, K),
                           T=DatasetLoader._load_parking_ground_truth(parking_path))

    @staticmethod
    def _load_kitti_ground_truth(path: str) -> np.ndarray:
        ground_truth_path = os.path.join(path, "poses", "05.txt")
        ground_truth = np.loadtxt(ground_truth_path)

        # load ground truth hom. transforms
        return np.array([np.append(np.reshape(T, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for T in ground_truth])

    @staticmethod
    def _load_kitti_frames(path: str, K: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        frame_path = os.path.join(path, "05", "image_0")
        frame_paths = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path) if
                              os.path.isfile(os.path.join(frame_path, f)) and f.endswith(".png")])
        for p in frame_paths:
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield K, img_gray

    @staticmethod
    def _load_malaga_frames(path: str, K: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        frame_path = os.path.join(path, "malaga-urban-dataset-extract-07_rectified_800x600_Images")
        frame_paths = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path) if
                              os.path.isfile(os.path.join(frame_path, f)) and f.endswith("left.jpg")])
        for p in frame_paths:
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield K, img_gray

    @staticmethod
    def _load_parking_ground_truth(path: str) -> np.ndarray:
        ground_truth_path = os.path.join(path, "poses.txt")
        ground_truth = np.loadtxt(ground_truth_path)
        # load ground truth hom. transforms
        return np.array([np.append(np.reshape(T, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for T in ground_truth])

    @staticmethod
    def _load_parking_frames(path: str, K: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        frame_path = os.path.join(path, "images")
        frame_paths = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path) if
                              os.path.isfile(os.path.join(frame_path, f)) and f.endswith(".png")])
        for p in frame_paths:
            img = cv2.imread(p)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield K, img_gray
