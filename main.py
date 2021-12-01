from utils.loadData import DatasetLoader, DatasetType
import cv2


def main():
    dataset = DatasetLoader(DatasetType.KITTI).load()

    for (K, frame) in dataset.frames:
        print(K)
        print(frame.shape)
        cv2.imshow("Image", frame)
        cv2.waitKey(0)

    ## cleanup
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
