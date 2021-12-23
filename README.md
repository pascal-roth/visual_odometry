# Project Vision for Mobile Robotics Projects

|                         |                    |
| ----------------------- | ------------------ |
| Required Python Version | `3.9`              |
| Deadline                | 9.1.2022, 23:59:59 |

## Datasets

| Name                        | URL                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| KITTI with IMU measurements | [kitti_IMU](https://polybox.ethz.ch/index.php/s/tqtLZ7wJstn0ooy) (1.4 GB)                         |
| kitti05                     | [kitti05](http://rpg.ifi.uzh.ch/docs/teaching/2021/kitti05.zip) (1.4 GB)                          |
| Malaga 07                   | [malaga07](http://rpg.ifi.uzh.ch/docs/teaching/2021/malaga-urban-dataset-extract-07.zip) (4.4 GB) |
| Parking garage              | [parking](http://rpg.ifi.uzh.ch/docs/teaching/2021/parking.zip) (208.3 MB)                        |

Download and extract to `/datasets`.

Vision Algorithm Project

1. Bootstrapping 2D <-> 2D point correspondences

- SIFT Feature Detector and Descriptor
  - Non- max suppression ?
- 5-point / 8-point Algorithm
  - RANSAC
  - Heuristic to choose the first two keyframes

2. Continuous VO

- Feature Detector SIFT
- Feature Matching
- P3P / RANSAC
  - get translation / rotation
- Heuristic when new landmarks are necessary
  - find some theory
  - try simple threshold (20% less feature matches than in the previous image)
- Bundle Ajdustment / Pose Graph Optimization
  - decide how often we need to do it?
  - matching with previous frames (parallel)

3. Bonus Feature

- Loop Closure via Bundle Adjustmen
  - current datasets don’t have loops!!
- alternative: scale drift reduction
  - need to find some papers describing how to do that
