from utils.loadData import DatasetLoader, DatasetType
from vo_pipeline.featureExtraction import FeatureExtractor, ExtractorType
from vo_pipeline.featureMatching import FeatureMatcher, MatcherType
import cv2


def matching_example(dataset):
    # feature descriptor and matching algorithm
    descriptor = FeatureExtractor(ExtractorType.SIFT)
    matcher = FeatureMatcher(MatcherType.BF)

    for (K, frame) in dataset.frames:
        [_, frame2] = next(dataset.frames)

        kp1, des1 = descriptor.get_kp(frame)
        kp2, des2 = descriptor.get_kp(frame2)

        matches = matcher.match_des(des1, des2)
        matcher.match_plotter(frame, kp1, frame2, kp2, matches)

        break


def main():
    dataset = DatasetLoader(DatasetType.KITTI).load()

    matching_example(dataset)

    for (K, frame) in dataset.frames:
        print(K)
        print(frame.shape)
        cv2.imshow("Image", frame)
        cv2.waitKey(0)

    ## cleanup
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
