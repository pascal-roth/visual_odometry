import cv2 as cv

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np

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
        points_proj  = self.K @ points_proj
        return points_proj[:, 0:2]


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
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    def bundle_adjustment(self, M: np.ndarray, landmarks: np.ndarray, camera_indices: np.ndarray, point_indices: np.ndarray, num_observations: int, keypoints: np.ndarray):
        """
        :param M: [num_observations, 3, 4] matrix containing the rotations and translations for each camera frame
        :param landmarks: 3D pointcloud
        :param camera_indices: contains the local frame indices for each observation
        :param point_indices: contains the local point indices for each observation
        :param num_observations: number of observations
        :param keypoints: set of keypoints detected in frames
        """
        # Number of landmarks for bundle adjustment
        n_points = landmarks.shape[0]
        # Here we save the translation and rotation vectors
        camera_params = np.zeros(num_observations, 6)
        camera_params[:, 3:6] = M[:, :, 3]
        # Transform from rot matrix to rot vector
        for i in range(num_observations):
            rot_vec, _ = cv.Rodrigues(M[i, :, :3])
            camera_params[i, :3] = rot_vec.ravel()

        x0 = np.hstack((camera_params.ravel(), landmarks.ravel()))
        # f0 = self._fun(x0, num_observations, n_points, camera_indices, point_indices, keypoints)
        A = self._bundle_adjustment_sparsity(num_observations, n_points, camera_indices, point_indices)
        res = least_squares(self._fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(num_observations, n_points, camera_indices, point_indices, keypoints))

        return res

