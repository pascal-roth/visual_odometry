import numpy as np

from queue import Queue
from utils.loadData import Dataset, DatasetLoader, DatasetType
from utils.message import IMUMessage, FeatureMessage, ImageMessage
from threading import Thread

from vio_pipeline.msckf import MSCKF
from vio_pipeline.image_processor import ImageProcessor


class VIO:
    def __init__(self, img_queue: 'Queue[ImageMessage]',
                 imu_queue: 'Queue[IMUMessage]',
                 dataset: Dataset):
        """Visual Innertail Odometry Pipeline

        Args:
            img_queue (Queue[ImageMessage]): input image queue, from dataset
            imu_queue (Queue[IMUMessage]): input IMU data queue, from dataset
        """
        self.K = dataset.K

        self.imu_queue: Queue[IMUMessage] = imu_queue
        self.img_queue: Queue[ImageMessage] = img_queue
        self.feature_queue: Queue[FeatureMessage] = Queue()

        self.image_processor = ImageProcessor(dataset.K, dataset.R_CAM_IMU, dataset.T_CAM_IMU)
        # self.msckf = MSCKF()

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            curr_frame = next(dataset.frames)
            if curr_frame is None:
                self.feature_queue.put(None)
                return

            feature_msg = self.image_processor.mono_callback(curr_frame)
            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            try:
                curr_imu = next(dataset.imu)

                self.image_processor.imu_callback(curr_imu)
                # TODO: fix msckf
                # self.msckf.imu_callback(curr_imu)
            except StopIteration:
                break

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            print(f"Feature Message: {feature_msg.timestamp}")
            # TODO: fix MSCKF
            # result = self.msckf.feature_callback(feature_msg)
            # TODO update viz


if __name__ == "__main__":

    dataset = DatasetLoader(DatasetType.KITTI_IMU).load()
    for i, frame_data in enumerate(dataset.frames):
        print(frame_data)
        if i == 10:
            break

    for i, imu_data in enumerate(dataset.imu):
        print(imu_data)
        if i == 10:
            break

    img_queue = Queue()
    imu_queue = Queue()
    vio = VIO(img_queue, imu_queue, dataset)
    # TODO
