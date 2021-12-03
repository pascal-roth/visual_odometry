# import packages
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
import multiprocessing
from typing import Callable, Tuple
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.bootstrap import BootstrapInitializer

class TrackPoints:
    
    def __init__(self, patch_radius: float, num_iters_GN: float, lambda_threshold: float):
        self.patch_radius = patch_radius
        self.num_iters_GN = num_iters_GN
        self.lambda_threshold = lambda_threshold

    def trackKLT(self, img0 : np.ndarray, img1: np.ndarray, keypoints: np.ndarray):
        """
        :param img0:                reference image from which we track the keypoints
        :param img1:                image to track keipoints in
        :param keypoints            keypoints to track
        :return                     returns keypoints located in img1        
        """

        N = keypoints.shape[0]
        
        keypoints1      = np.zeros((N, 2))
        pointcloud_mask = np.full((N, 1), False, dtype=bool)
        W = np.zeros((2, 3))
        
        # pool = multiprocessing.Pool(processes=8)
        # def my_function(img0, img1, keypoints):
        #     W               = self._track(img0, img1, keypoints)
        #     kp_candidate    = keypoints + W[:, 2]
        #     W_reverse       = self._track(img1, img0, kp_candidate)
        #     kp_check        = kp_candidate - W_reverse[:, 2]

        #     if self._bidirectionalTest(kp_candidate, kp_check):
        #         keypoints1[n,:] = kp_candidate
        #         return kp_candidate, True
        #     return np.array([0, 0]), False
        
        # keypoints1[:] , pointcloud_mask[:] = pool.map(my_function, [img0, img1, (keypoints[n,:] for n in range(N))] )
        
        # keypoints1 = keypoints1[~np.all(keypoints1 == 0, axis=1)]
        
        
            

        for n in range(N):
            
            W               = self._track(img0, img1, keypoints[n,:])
            kp_candidate    = keypoints[n, :] + W[:, 2]
            W_reverse       = self._track(img1, img0, kp_candidate)
            kp_check        = kp_candidate - W_reverse[:, 2]

            if self._bidirectionalTest(kp_candidate, kp_check):
                keypoints1[n,:] = np.vstack((keypoints1, np.expand_dims(kp_candidate, axis=0)))
                pointcloud_mask[n, :] = True
            
        
        return keypoints1, pointcloud_mask
    
    def _track(self, img0 : np.ndarray, img1 : np.ndarray, keypoint : np.ndarray) -> np.ndarray:
        """
        :param img0:                reference image from which we track the keypoints
        :param img1:                image to track keipoints in
        :keypoint:                  keypoint form img1 to be tracked
        :return                     W, affine transformation
        """
        
        # Define warp and its parameters. Start with identity transformation
        p = np.array([1, 0, 0, 1, 0, 0])
        W = np.reshape(p, (2, 3), order='F')
        # Get patch of reference image
        patch_img0 = self._getWarpedPatch(img0, W, keypoint, self.patch_radius)
        # Vectorize to get iR
        iR = np.reshape(patch_img0, (patch_img0.size, 1))
        
        
        # Define derivatives that will be useful in the algorithm
        dw_dp = np.zeros((2, 6))
        dw_dp[:, 4:6] = np.eye(2)
        di_dp = np.zeros(((2 * self.patch_radius + 1)**2, 6))
        
        # Apply the Gauss-Newton algorithm to get W
        for n in range(self.num_iters_GN):
            # Get patch from img1 using current estimated W
            patch_img1 = self._getWarpedPatch(img1, W, keypoint, self.patch_radius) 
            # Vectorize to get i
            i = np.reshape(patch_img1, (patch_img1.size, 1))
            
            # Compute gradients wrt to I and vectorize
            patch_grad = self._getWarpedPatch(img1, W, keypoint, self.patch_radius + 1)
            Ix = convolve2d(patch_grad[1:-1, :], np.array([[1, 0, -1]]), boundary='symm', mode='valid')
            Iy = convolve2d(patch_grad[:, 1:-1], np.array([[1, 0, -1]]).T, boundary='symm', mode='valid')
            ix = np.reshape(Ix, (Ix.size, 1))
            iy = np.reshape(Iy, (Iy.size, 1))
            
            # Compute di_dp
            m = 0
            for x in range(-self.patch_radius, self.patch_radius):
                for y in range(-self.patch_radius, self.patch_radius):
                    dw_dp[:, 0:2] = np.eye(2) * x
                    dw_dp[:, 2:4] = np.eye(2) * y
                    di_dp[m, :] = np.array([ix[m][0], iy[m][0]]) @ dw_dp
                    m += 1
            
            # Compute hessian
            H = di_dp.T @ di_dp
            # Compute delta_p
            delta_p = np.linalg.inv(H) @ di_dp.T @ (iR - i)
            # Apply gradient update
            p = p + delta_p
            W = np.reshape(W, (2, 3), order='F')  
            
        return W          
        

    def _getWarpedPatch(self, img : np.ndarray, W : np.ndarray, kp : np.ndarray, r_T : int) -> np.ndarray:
        """
        :img:                       image to get the warp from
        :W                          transformation matrix
        :kp:                        keypoint form img1 to be tracked
        :return                     patch_warped, transformed patched
        """

        xc = kp[0]
        yc = kp[1]

        V  = np.zeros((2,r_T*2+1))
        k  = 0
        
        # Iterate over pixels of the patch
        for x in range(-r_T,r_T):
            for y in range(-r_T,r_T):
                # Homogeneous coordinates
                v  = np.array([[x],[y],[1]])
                # Frame transformation
                v_ = W @ v
                # Store new coordinates
                V[:,k:k+1] = v_
                k += 1

        # Bilinear interpolation
        f = interp2d(np.arange(img.shape[1]),np.arange(img.shape[0]),np.reshape(img,(1,-1))[0],'linear')
        P = f(V[0,:]+xc,V[1,:]+yc)
        P = np.nan_to_num(P,nan=0)

        return P
        
    def _bidirectionalTest(self, kp0 : np.ndarray, kp_reprojected : np.ndarray) -> bool:
        """
        :param kp0:                 original keypoint from reference image
        :param kp_reprojected:      reprojected keypoint using found transformation
        :return                     bool with result of test
        """

        return np.all(np.abs(kp0-kp_reprojected) < self.lambda_threshold)