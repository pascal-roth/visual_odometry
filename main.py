from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.bootstrap import BootstrapInitializer
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
    dataset = DatasetLoader(DatasetType.KITTI).load()
    K, img0 = next(dataset.frames)
    next(dataset.frames)
    next(dataset.frames)
    _, img1 = next(dataset.frames)
    
    bootstrapper = BootstrapInitializer(img0, img1, K)

    pts = bootstrapper.point_cloud
    T = bootstrapper.T

    # compute some error metrics compared to ground truth transform
    t_hat = T[0:3, 3]
    t_gt = dataset.T[3, 0:3, 3]
    angle = np.arccos(np.dot(t_hat, t_gt) / (np.linalg.norm(t_hat) * np.linalg.norm(t_gt)))
    R_hat = T[0:3, 0:3]
    R_gt = dataset.T[3, 0:3, 0:3]
    print("Error metrics:")
    print(f"t_err angle = {np.rad2deg(angle)} deg, R_err frob. norm {np.linalg.norm(R_hat - R_gt)}")

    # plot resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label="reconstructed points")
    ax.scatter(0, 0, 0, color="red", label="$t_0$")
    t_3 = -T[0:3, 0:3].T @ T[0:3, 3]
    t_3 =  T[0:3, 3]
    tgt_3 =  dataset.T[3, 0:3, 3]
    ax.scatter(t_3[0], t_3[1], t_3[2], color="yellow", label="$t_3$")
    ax.scatter(tgt_3[0], tgt_3[1], tgt_3[2], color="green", label="$t^{gt}_3$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.title("Reconstructed point cloud")
    plt.show()


def main():
    matching_example()
    bootstraping_example()


if __name__ == "__main__":
    main()
