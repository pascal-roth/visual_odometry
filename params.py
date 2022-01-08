# General

# Used dataset: KITTI, MALAGA, PARKING
DATASET = "PARKING"

# Queue size used in BA
FRAME_QUEUE_SIZE = 100

# Bootstrapping phase
# Feature matching
MATCHING_THRESHOLD: float = 0.6

# RANSAC Fundamental matrix estimation
RANSAC_REPROJ_THRESHOLD: float = .5
RANSAC_CONFIDENCE: float = 0.999
RANSAC_MAX_ITERS: int = 1000

# KLT Parameters
KLT_RADIUS = 21
KLT_LAMBDA: float = 0.1
KLT_N_ITERS: int = 100
KLT_NUM_PYRAMIDS = 8
KLT_MIN_EIGEN_THRESHOLD = 1e-3

MAX_BASELINE_UNCERTAINTY = 0.05

BA_DISTANCE_TH = 1e2
