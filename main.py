from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
from vo_pipeline.bootstrap import BootstrapInitializer
from utils.matrix import *
import numpy as np
import cv2


def matching_example(dataset):
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


def main():
    dataset = DatasetLoader(DatasetType.KITTI).load()

    # matching_example(dataset)

    img0 = None
    img1 = None
    for i, frame in enumerate(dataset.frames):
        K = frame[0]
        if i == 0:
            img0 = frame[1]
        elif i == 3:
            img1 = frame[1]
            break

    bootstraper = BootstrapInitializer(img0, img1, K)
    T = bootstraper.T
    print(T)
    print(dataset.T[3])

    t_hat = T[0:3, 3]
    t_gt = dataset.T[3, 0:3, 3]
    angle = np.arccos(np.dot(t_hat, t_gt) / (np.linalg.norm(t_hat) * np.linalg.norm(t_gt)))
    R_hat = T[0:3, 0:3]
    R_gt = dataset.T[3, 0:3, 0:3]
    print("Error metrics:")
    print(f"t_err angle = {np.rad2deg(angle)}, R_err frob. norm {np.linalg.norm(R_hat - R_gt)}")


if __name__ == "__main__":
    main()
