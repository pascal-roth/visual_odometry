import threading
from queue import Queue
from threading import Thread
import sys
from typing import List, Tuple

import numpy as np
from params import *

from utils.message import FeatureData, FeatureMeasurement, LandmarkData, PoseData
from utils.transform import HomTransform

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


class Viewer(object):
    def __init__(self):
        self.pose_queue = Queue()
        # keep at most the 10 past frames in memory
        self.image_queue = Queue(maxsize=10)
        self.feature_queue = Queue()
        self.landmark_queue = Queue()

    def update_pose(self, pose: HomTransform):
        if pose is None:
            return
        self.pose_queue.put(pose.to_matrix())

    def update_image(self, image: np.ndarray):
        if image is None:
            return
        self.image_queue.put(image)

    def update_features(self, features: FeatureData):
        if features is None:
            return
        self.feature_queue.put(features)

    def update_landmarks(self, landmarks: LandmarkData):
        if landmarks is None:
            return
        self.landmark_queue.put(landmarks)

    def start_vis(self):
        # Create figure to draw on
        figure = plt.figure()

        # Image subplot with landmark and keypoint overlays
        ax_img = figure.add_subplot(221)
        image_show = ax_img.imshow(np.full((370, 1226), 128, dtype=np.uint8),
                                   interpolation="none",
                                   cmap='gray',
                                   animated=True)
        ax_img.set_title("Current Image")
        sc_landmarks = ax_img.scatter([], [],
                                      s=10,
                                      color="green",
                                      marker="x",
                                      label="landmarks")
        sc_keypoints = ax_img.scatter([], [],
                                      s=10,
                                      color="red",
                                      marker="x",
                                      label="keypoints")
        ax_img.legend()

        # Full trajectory subplot
        ax_full_traj = figure.add_subplot(246)
        ax_full_traj.set_title("Full trajectory")
        sc_full_traj, = ax_full_traj.plot([], [],
                                          color="blue",
                                          label="trajectory",
                                          lw=2)

        # Number if matched landmarkim.set_data(np.random.random((5, 5)))s
        ax_tracked_kps = figure.add_subplot(245)
        ax_tracked_kps.set_title("# tracked landmarks over last 20 frames")
        sc_tracked_kps, = ax_tracked_kps.plot([], [],
                                              color='red',
                                              label="matched lks",
                                              lw=2)

        # Trajectory of last 20 frames and landmarks subplot
        ax_local_traj = figure.add_subplot(122)
        ax_local_traj.set_title("Trajectory of last 20 frames and landmarks")
        sc_local_traj = ax_local_traj.scatter([], [],
                                              s=5,
                                              color="blue",
                                              marker="o",
                                              label="trajectory")
        sc_local_lks = ax_local_traj.scatter([], [],
                                             s=3,
                                             color="black",
                                             marker="*",
                                             label="landmarks")

        traj_list: List[np.ndarray] = []
        num_landmarks: List[Tuple[float, int]] = []

        # update function, run at every tick of self.timer
        def update(i):
            x_idx = 0
            y_idx = 2

            # Update image plot
            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    img_data = self.image_queue.get()
                image_show.set_array(img_data)

            # plot full trajectory
            while not self.pose_queue.empty():
                traj_list.append(self.pose_queue.get())

            trajectory = np.array([pose[:3, 3] for pose in traj_list])
            if trajectory.size > 0:
                sc_full_traj.set_data(trajectory[:, x_idx], trajectory[:,
                                                                       y_idx])

                xy_min = np.min(trajectory, axis=0)
                xy_max = np.max(trajectory, axis=0)
                ax_full_traj.set_xlim(xy_min[x_idx] - 2, xy_max[x_idx] + 2)
                ax_full_traj.set_ylim(xy_min[y_idx] - 2, xy_max[y_idx] + 2)

            # plot last 20 frames trajectory and landmarks
            landmark = None
            landmarks = None
            while not self.landmark_queue.empty():
                landmark = self.landmark_queue.get()
                landmarks = landmark.landmarks

            if trajectory.size > 0:
                sc_local_traj.set_offsets(trajectory[-20:, [x_idx, y_idx]])
            if landmarks is not None and landmarks.size > 0:
                landmarks = landmarks[:, [x_idx, y_idx]]
                # plot landmarks and last 20 trajectory poses
                sc_local_lks.set_offsets(landmarks)

                # set plot limits to show all landmarks
                all_active = np.vstack((landmarks, trajectory[-20:,
                                                              [x_idx, y_idx]]))
                xy_min = np.min(all_active, axis=0)
                xy_max = np.max(all_active, axis=0)
                ax_local_traj.set_xlim(xy_min[0] - 2, xy_max[0] + 2)
                ax_local_traj.set_ylim(xy_min[1] - 2, xy_max[1] + 2)

            # plot number of tracked landmarks
            if landmark is not None:
                num_landmarks.append((landmark.timestamp, len(landmarks)))
            if len(num_landmarks) > 0:
                last_landmarks = np.array(num_landmarks)[-20:]
                start_time = np.min(last_landmarks[:, 0])
                sc_tracked_kps.set_data(last_landmarks[:, 0] - start_time, last_landmarks[:, 1])
            return image_show,

        ani = animation.FuncAnimation(figure,
                                      update,
                                      blit=False,
                                      interval=1000 / TARGET_FRAMERATE)
        figure.tight_layout()
        plt.show()
