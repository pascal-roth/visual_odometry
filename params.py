# General
from re import A

FRAME_QUEUE_SIZE = 100

# Bootstrapping phase
# Feature matching
MATCHING_THRESHOLD: float = 0.6

# RANSAC Fundamental matrix estimation
RANSAC_REPROJ_THRESHOLD: float = .5
RANSAC_CONFIDENCE: float = 0.999
RANSAC_MAX_ITERS: int = 1000

# KLT Parameters
KLT_RADIUS = 15
KLT_LAMBDA: float = 0.1
KLT_N_ITERS: int = 100
KLT_NUM_PYRAMIDS = 3
KLT_MIN_EIGEN_THRESHOLD = 1e-3
KLT_TRACK_PRECISION = 0.01

MAX_BASELINE_UNCERTAINTY = 0.1

BA_DISTANCE_TH = 1e2

#
# IMU Params:
#

# Number of samples to use for estimating the gravity vector
GRAVITY_ESTIMATION_SAMPLES: int = 50
CHI2_CONFIDENCE = 0.95

# Initial covariance of orientation and position
VELOCITY_COV = 0.25
GYRO_BIAS_COV = 0.01
ACC_BIAS_COV = 0.01
EXTRINSIC_ROTATION_COV = 3.0462e-4
EXTRINSIC_TRANSLATION_COV = 2.5e-5

# Noise related parameters (Variances)
GYRO_NOISE = 0.005**2
ACC_NOISE = 0.05**2
GYRO_BIAS_NOISE = 0.001**2
ACC_BIAS_NOISE = 0.01**2
OBSERVATION_NOISE = 10**2

POSITION_DELTA_THRESHOLD = 1
VELOCITY_DELTA_THRESHOLD = .5

# Grid parameters
# images have aspect ratio ~3.31
GRID_RESOLUTION = 1
GRID_ROW = 4 * GRID_RESOLUTION
GRID_COL = 13 * GRID_RESOLUTION
GRID_NUM = GRID_COL * GRID_ROW
GRID_MIN_FEATURE = 3
GRID_MAX_FEATURE = 5

#
# DISPLAY params:
#
TARGET_FRAMERATE = 10

# Maximum number of camera states to be stored
MAX_CAM_STATE_SIZE = 20

# The position uncertainty threshold is used to determine
# when to reset the system online. Otherwise, the ever-increaseing
# uncertainty will make the estimation unstable.
# Note this online reset will be some dead-reckoning.
# Set this threshold to nonpositive to disable online reset.
POSITION_STD_THRESHOLD = 5

# Each dataset stream writes it's data into a threadsafe queue.
# The size of this queue determines the maximal number of data samples
# to load eagerly, before they are used.
# (10 times this value for IMU data)
DATASET_FRAME_QUEUE_SIZE = 5


#
# Optimization config
#
class OptimizationParams(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self.translation_threshold = 0.2  # 0.2
        self.huber_epsilon = 0.01
        self.estimation_precision = 5e-7
        self.initial_damping = 1e-3
        self.outer_loop_max_iteration = 10  # 10
        self.inner_loop_max_iteration = 10  # 10
