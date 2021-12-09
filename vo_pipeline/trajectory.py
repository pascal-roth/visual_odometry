import numpy as np

class Trajectory:
    def __init__(self, traj_idx: int, init_frame_idx: int, pt: np.ndarray,
                 transform: np.ndarray):
        """Trajectory of a keypoint tracked over some number of frames

        Args:
            traj_idx (int): id of this trajectory
            init_frame_idx (int): index of initial frame
            pt (np.ndarray): 2D image coordinates of initial point
            transform (np.ndarray): homogeneous transformation from world coordinate to camera coordinates
        """
        self.traj_idx = traj_idx

        self.init_idx = init_frame_idx
        self.init_point = pt
        self.init_transform = np.float32(transform)

        self.final_idx: int = init_frame_idx
        self.final_point: np.ndarray = pt
        self.final_transform: np.ndarray = np.float32(transform)

    def tracked_to(self, frame_idx: int, pt: np.ndarray, transform: np.ndarray):
        """Update the last point this trajectory could be tracked to

        Args:
            frame_idx (int): final frame idx
            pt (np.ndarray): 2D image coordinates of the final point
            transform (np.ndarray): homogeneous transformation from world coordinate to camera coordinates
        """
        self.final_idx = frame_idx
        self.final_point = pt
        self.final_transform = np.float32(transform)
