from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.bootstrap import BootstrapInitializer
from vo_pipeline.poseEstimation import PoseEstimation
import matplotlib.pyplot as plt
from utils.matrix import *
import numpy as np
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
    dataset = DatasetLoader(DatasetType.PARKING).load()
    _, img0 = next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    K, img1 = next(dataset.frames)
    i = 5

    bootstrapper = BootstrapInitializer(img0, img1, K)

    pts = bootstrapper.point_cloud
    T = bootstrapper.T

    # plot resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label="reconstructed points")
    ax.scatter(0, 0, 0, "*", color="red", label="$t_0$")
    t_3 = T[0:3, 3]
    t_gt = dataset.T[i, 0:3, 3]
    print(f"t_3: {t_3}, t_gt: {t_gt}")
    ax.scatter(t_3[0], t_3[1], t_3[2], "*", color="yellow", label=f"$t_{i}$")
    ax.scatter(t_gt[0], t_gt[1], t_gt[2], "*", color="green", label=f"$tgt_{i}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.title("Reconstructed point cloud")
    plt.show()


def main():
    # matching_example()
    bootstraping_example()


if __name__ == "__main__":
    main()
