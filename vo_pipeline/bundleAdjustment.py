import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np
from typing import Tuple
import params


class BundleAdjustment:

    def __init__(self, K: np.ndarray):
        self.K = K

    def _rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def _reproject(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self._rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        points_proj = self.K @ points_proj.T
        points_proj = points_proj.T
        return points_proj[:, :2]/points_proj[:, 2, None]

    def _fun(self, params, n_frames, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_frames * 6].reshape((n_frames, 6))
        points_3d = params[n_frames * 6:].reshape((n_points, 3))
        points_proj = self._reproject(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def _bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)

        start_index = np.count_nonzero(camera_indices == 0)
        # start_index = 0
        for s in range(6):
            A[2 * i[start_index:], camera_indices[start_index:] * 6 + s] = 1
            A[2 * i[start_index:] + 1, camera_indices[start_index:] * 6 + s] = 1

        for s in range(3):
            A[2 * i[start_index:], n_cameras * 6 + point_indices[start_index:] * 3 + s] = 1
            A[2 * i[start_index:] + 1, n_cameras * 6 + point_indices[start_index:] * 3 + s] = 1

        return A

    def bundle_adjustment(self, M: np.ndarray, landmarks: np.ndarray, camera_indices: np.ndarray,
                          point_indices: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param M: [num_frames, 3, 4] matrix containing the rotations and translations for each camera frame
        :param landmarks: 3D pointcloud
        :param camera_indices: contains the local frame indices for each observation [num_frames * num_kp[i], 1] kp_idx -> frame_idx [0, num_frames - 1]
        :param point_indices: contains the local point indices for each observation [num_frames * num_kp[i]] kp_imp -> landmark_idx
        :param keypoints: set of keypoints detected in frames [num_frames * num_kp[i], 2]

        :return Tuple[adjusted landmarks of pointcloud [n_landmarks, 3], adjusted camera poses M]
        """

        keypoints, camera_indices, point_indices = self._preprocess_data(M, landmarks, camera_indices, point_indices, keypoints)
        # Number of landmarks for bundle adjustment
        n_points = landmarks.shape[0]
        # Here we save the translation and rotation vectors
        num_frames = M.shape[0]
        camera_params = np.zeros((num_frames, 6))
        camera_params[:, 3:6] = M[:, :, 3]
        # Transform from rot matrix to rot vector
        for i in range(num_frames):
            rot_vec, _ = cv.Rodrigues(M[i, :, :3])
            camera_params[i, :3] = rot_vec.ravel()

        x0 = np.hstack((camera_params.ravel(), landmarks.ravel()))
        # f0 = self._fun(x0, num_frames, n_points, camera_indices, point_indices, keypoints)
        A = self._bundle_adjustment_sparsity(num_frames, n_points, camera_indices, point_indices)
        res = least_squares(self._fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(num_frames, n_points, camera_indices, point_indices, keypoints))

        # Postprocessing of solution
        adjusted_landmarks = res.x[camera_params.ravel().shape[0]:].reshape(landmarks.shape)
        adjusted_rot_vec = res.x[:camera_params.ravel().shape[0]].reshape(camera_params.shape)[:, :3]
        adjusted_trans = res.x[:camera_params.ravel().shape[0]].reshape(camera_params.shape)[:, 3:]
        adjusted_M = np.zeros(M.shape)
        adjusted_M[:, :, 3] = adjusted_trans
        for i in range(num_frames):
            R, _ = cv.Rodrigues(adjusted_rot_vec[i, :])
            adjusted_M[i, :, :3] = R

        return adjusted_M, adjusted_landmarks

    def _preprocess_data(self, M: np.ndarray, landmarks: np.ndarray, camera_indices: np.ndarray,
                          point_indices: np.ndarray, keypoints: np.ndarray):
        n_points = landmarks.shape[0]
        # Here we save the translation and rotation vectors
        num_frames = M.shape[0]
        camera_params = np.zeros((num_frames, 6))
        camera_params[:, 3:6] = M[:, :, 3]
        # Transform from rot matrix to rot vector
        for i in range(num_frames):
            rot_vec, _ = cv.Rodrigues(M[i, :, :3])
            camera_params[i, :3] = rot_vec.ravel()

        x0 = np.hstack((camera_params.ravel(), landmarks.ravel()))

        difference = self._fun(x0, num_frames, n_points, camera_indices, point_indices, keypoints)**2
        difference = difference.reshape((int(difference.shape[0] / 2), 2))
        difference = np.sum(difference, axis=1)
        outliers = difference > params.BA_DISTANCE_TH

        keypoints = np.delete(keypoints, outliers, axis=0)
        point_indices = np.delete(point_indices, outliers, axis=0)
        camera_indices = np.delete(camera_indices, outliers, axis=0)

        return keypoints, camera_indices, point_indices







