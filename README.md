# Project Vision for Mobile Robotics Projects

|                         |                    |
| ----------------------- | ------------------ |
| Required Python Version | `3.9`              |
| Deadline                | 9.1.2022, 23:59:59 |

## Datasets

| Name                        | URL                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| KITTI with IMU measurements | [kitti_IMU_v3](https://polybox.ethz.ch/index.php/s/b3aGsvAY22wcdOk) (1.3 GB)                      |

Download and extract to `/datasets`.


## Running the pipeline

1) Download the above datasets and extract them to `/datasets` with the directory names:
`kitti_IMU`.
2) Choose the desired dataset in the `params.py` file.
3) Run `python main.py`.
4) Enjoy our VO pipeline :)

## Computer characteristics

- CPU frequency: 2.8 GHz
- RAM: 16GB
- Number of used threads: 1 for VO, 6 for VIO


