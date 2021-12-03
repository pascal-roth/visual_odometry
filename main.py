from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.bootstrap import BootstrapInitializer
import matplotlib.pyplot as plt
from utils.matrix import *
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

    # CAUTION: to get t_i in world frame we need to invert the W -> C2 transformation T
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


def main():
    logging.basicConfig(level=logging.INFO)
    # matching_example()
    bootstraping_example()


if __name__ == "__main__":
    main()
