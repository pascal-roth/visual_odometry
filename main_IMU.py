from queue import Queue
from utils.loadData import Dataset, DatasetLoader, DatasetType
from utils.message import IMUMessage, FeatureMessage, ImageMessage
from threading import Thread

from vio_pipeline.msckf import MSCKF


class VIO:
    def __init__(self, img_queue: 'Queue[ImageMessage]',
                 imu_queue: 'Queue[IMUMessage]'):
        """Visual Innertail Odometry Pipeline

        Args:
            img_queue (Queue[ImageMessage]): input image queue, from dataset
            imu_queue (Queue[IMUMessage]): input IMU data queue, from dataset
        """
        self.imu_queue: Queue[IMUMessage] = imu_queue
        self.img_queue: Queue[ImageMessage] = img_queue
        self.feature_queue: Queue[FeatureMessage] = Queue()

        self.image_processor = None
        self.msckf = MSCKF()

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # TODO

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # TODO process IMU data in feature processor
            # self.
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            print(f"Feature Message: {feature_msg.timestamp}")
            result = self.msckf.feature_callback(feature_msg)
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
    vio = VIO()
    # TODO
