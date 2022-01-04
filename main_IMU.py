import numpy as np

from queue import Queue
import queue
from utils.loadData import Dataset, DatasetLoader, DatasetType, DataPublisher
from utils.message import IMUData, FeatureData, FrameData
from threading import Thread
import logging
from vispy import app

from vio_pipeline.msckf import MSCKF
from vio_pipeline.image_processor import ImageProcessor
from vio_pipeline.viewer import Viewer
from params import DATASET_FRAME_QUEUE_SIZE


class VIO:
    def __init__(self, img_queue: 'Queue[FrameData]',
                 imu_queue: 'Queue[IMUData]', dataset: Dataset,
                 viewer: Viewer):
        """Visual Innertail Odometry Pipeline

        Args:
            img_queue (Queue[ImageMessage]): input image queue, from dataset
            imu_queue (Queue[IMUMessage]): input IMU data queue, from dataset
        """
        self.K = dataset.K

        self.imu_queue: Queue[IMUData] = imu_queue
        self.img_queue: Queue[FrameData] = img_queue
        # keep feature queue small, since it's consumer is much slower than the producer
        self.feature_queue: Queue[FeatureData] = Queue(maxsize=2)
        self.image_processor = ImageProcessor(dataset.K, dataset.R_CAM_IMU,
                                              dataset.T_CAM_IMU)
        self.msckf = MSCKF(dataset.R_CAM_IMU)
        self.viewer = viewer
        # IMPORTANT: !!! Any parameters accessed by any of the threads
        # has to be declared before this point !!!

        self.img_thread = Thread(target=self.process_img, name="Image Thread")
        self.imu_thread = Thread(target=self.process_imu, name="IMU Thread")
        self.feature_thread = Thread(target=self.process_feature,
                                     name="Feature Thead")
        self.img_thread.start()
        self.imu_thread.start()
        self.feature_thread.start()

    def process_img(self):
        print("Started image processing thread")
        while True:
            curr_frame = self.img_queue.get()
            if curr_frame is None:
                # abort image processing
                self.feature_queue.put(None)
                return

            feature_msg = self.image_processor.mono_callback(curr_frame)
            if feature_msg is not None:
                self.feature_queue.put(feature_msg)
            if self.viewer is not None:
                self.viewer.update_image(curr_frame.image)

    def process_imu(self):
        print("Started imu processing thread")
        while True:
            try:
                curr_imu = self.imu_queue.get()
                if curr_imu is None:
                    return

                self.image_processor.imu_callback(curr_imu)
                self.msckf.imu_callback(curr_imu)
            except StopIteration:
                break

    def process_feature(self):
        print("Started feature processing thread")
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            # print(f"Feature Message: {feature_msg.timestamp}")
            result = self.msckf.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print(f"Loading dataset: {DatasetType.KITTI_IMU}")
    dataset = DatasetLoader(DatasetType.KITTI_IMU).load()

    frame_queue = Queue(maxsize=DATASET_FRAME_QUEUE_SIZE)
    imu_queue = Queue(maxsize=20 * DATASET_FRAME_QUEUE_SIZE)
    frame_publisher = DataPublisher(dataset.frames,
                                    frame_queue,
                                    type_name="Frame")
    img_publisher = DataPublisher(dataset.imu, imu_queue, type_name="IMU")
    # start publishers
    frame_publisher.start()
    img_publisher.start()

    # create viewer on main thread, since otherwise vispy does not work
    viewer = Viewer()

    vio = VIO(frame_queue, imu_queue, dataset, viewer)

    if sys.flags.interactive == 0:
        app.run()
