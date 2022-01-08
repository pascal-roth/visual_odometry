import copy
import logging

import params
from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.poseEstimation import AlgoMethod, PoseEstimation
from vo_pipeline.continuousVO import ContinuousVO
import matplotlib.pyplot as plt
from utils.matrix import *
from utils.plotter import *
import matplotlib.animation as animation
import numpy as np
import logging
import cv2



def matching_example():
    dataset = DatasetLoader(DatasetType.KITTI).load()
    # feature descriptor and matching algorithm
    descriptor = FeatureExtractor(ExtractorType.SIFT)
    matcher = FeatureMatcher(MatcherType.BF)

    for (K, frame) in dataset.frames:
        [_, frame2] = next(dataset.frames)

        kp1, des1 = descriptor.get_kp(frame)
        kp2, des2 = descriptor.get_kp(frame2)

        matches = matcher.match_descriptors(des1, des2)
        matcher.match_plotter(frame, kp1, frame2, kp2, matches)
        break


def bootstraping_example():
    dataset = DatasetLoader(DatasetType.KITTI).load()
    _, img0 = next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    K, img1 = next(dataset.frames)
    i = 4
    # img0 = cv2.imread("./0001.jpg")
    # img1 = cv2.imread("./0002.jpg")
    #
    # K = np.array([[1379.74, 0, 760.35],
    #               [0, 1382.08, 503.41],
    #               [0, 0, 1]])

    bootstrapper = BootstrapInitializer(img0, img1, K, max_point_dist=50)

    pts = bootstrapper.point_cloud
    T = bootstrapper.T

    R_C2_W = T[0:3, 0:3]
    T_C2_W = T[0:3, 3]

    # CAUTION: to get t_i in world frame we need to invert the W -> Ci transformation T
    t_i_W = -R_C2_W.T @ T_C2_W
    t_gt_W = dataset.T[i, 0:3, 3]
    R_gt_W = dataset.T[3, 0:3, 0:3]

    logging.info("Reconstruction successful!")
    logging.info(f"t_3: {t_i_W}, t_gt: {t_gt_W}")
    # compute some error metrics
    angle = np.arccos(np.dot(t_i_W, t_gt_W) / (np.linalg.norm(t_i_W) * np.linalg.norm(t_gt_W)))
    logging.info("Error metrics:")
    logging.info(
        f"t_err angle = {np.rad2deg(angle)} deg, t_err abs dist: {np.linalg.norm(t_i_W - t_gt_W)}, R_err frob. norm {np.linalg.norm(R_C2_W - R_gt_W)}")

    # plot resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label="reconstructed points")
    ax.scatter(0, 0, 0, "*", color="red", label="$t_0$")
    ax.scatter(t_i_W[0], t_i_W[1], t_i_W[2], "*", color="yellow", label=f"$t_{i}$")
    ax.scatter(t_gt_W[0], t_gt_W[1], t_gt_W[2], "*", color="green", label=f"$tgt_{i}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.title("Reconstructed point cloud")
    plt.show()


def poseEstimation_example():
    dataset = DatasetLoader(DatasetType.KITTI).load()

    M = []

    _, img1 = next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    K, img2 = next(dataset.frames)
    i = 4


    bootstrapper = BootstrapInitializer(img1, img2, K, max_point_dist=50)
    pointcloud = bootstrapper.point_cloud
    poseEstimator = PoseEstimation(K, pointcloud[:,0:3], bootstrapper.pts2[:, 0:2], use_KLT=True, algo_method_type=AlgoMethod.P3P)

    # plot resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], label="reconstructed points")
    ax.scatter(0, 0, 0, "*", color="red", label="$t_0$")

    def hom_inv(T):
        I = np.zeros_like(T)
        I[0:3, 0:3] = T[0:3, 0:3].T
        I[0:3, 3] = -T[0:3, 0:3].T @ T[0:3, 3]
        return I
    
    pose_init = hom_inv(bootstrapper.T)
    t_act = pose_init[0:3, 3]
    gt_scale = np.linalg.norm(t_act) / np.linalg.norm(dataset.T[i, 0:3, 3])
    ax.scatter(t_act[0], t_act[1], t_act[2], "*", color="yellow", label=f"$t_{4}$")
    print(f"t_act: {t_act}")

    prev_img_kpts = bootstrapper.pts2[:, 0:2]
    prev_img = img2
    for idx in range(15):

        # extract feature in images
        _, img = next(dataset.frames)
        prev_img_kpts = poseEstimator.match_key_points(pointcloud[:, 0:3], prev_img_kpts, bootstrapper.pts_des2,
                                                                      prev_img, img)
        M.append(poseEstimator.PnP(prev_img_kpts))
        prev_img = copy.copy(img)
        # Maybe np.linalg.inv(M[idx])
        pose = pose_init @ hom_inv(M[idx])
        t_act = pose[:, 3]
        ax.scatter(t_act[0], t_act[1], t_act[2], "*", color="red")
        t_gt = gt_scale * dataset.T[i + idx, 0:3, 3]
        ax.scatter(t_gt[0], t_gt[1], t_gt[2], "*", color="green")
        print(f"t_act: {t_act}")
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.title("Reconstructed point cloud")
    plt.show()


def continuous_vo_example():

    max_point_distance = 50
    if params.DATASET == "KITTI":
        dataset_type = DatasetType.KITTI
    elif params.DATASET == "PARKING":
        dataset_type = DatasetType.PARKING
        max_point_distance = 200
    elif params.DATASET == "MALAGA":
        dataset_type = DatasetType.MALAGA
    else:
        raise NameError('WRONG DATASET NAME!')
    dataset = DatasetLoader(dataset_type).load()
    continuousVO = ContinuousVO(dataset, frame_queue_size=250, max_point_distance=max_point_distance)

    # plt_groud_truth(dataset)
    # plt_only_trajectory(continuousVO, dataset)
    plt_online(continuousVO, dataset)


def main():
    logging.basicConfig(level=logging.INFO)
    # matching_example()
    # bootstraping_example()
    # poseEstimation_example()
    continuous_vo_example()


if __name__ == "__main__":
    main()
