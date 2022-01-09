# General

# Used dataset: KITTI, MALAGA, PARKING
DATASET = "KITTI"

# Queue size used in BA
FRAME_QUEUE_SIZE = 100

# Feature descriptors

# Bootstrapping phase
# Feature matching
# minimal distance ratio the second closest point has to be away from a match
MATCHING_THRESHOLD: float = 0.8
MATCHING_RATIO = 0.4
MIN_FRAME_DIST = 5

# Bootstrapping phase
#
# RANSAC Fundamental matrix estimation
RANSAC_REPROJ_THRESHOLD: float = .5 if DATASET == 'PARKING' else .1
RANSAC_CONFIDENCE: float = 0.999 if DATASET == 'PARKING' else 0.9999
RANSAC_MAX_ITERS: int = 10000 if DATASET == 'PARKING' else 100

# PnP
PNP_RANSAC_REPROJ_THRESHOLD: float = 1
PNP_RANSAC_CONFIDENCE: float = 0.99 if DATASET == 'PARKING' else 0.999
PNP_RANSAC_MAX_ITERS: int = 10000 if DATASET == 'PARKING' else 100

# KLT Parameters
KLT_RADIUS = 21
KLT_LAMBDA: float = 0.1
KLT_N_ITERS: int = 10
KLT_NUM_PYRAMIDS = 8
KLT_MIN_EIGEN_THRESHOLD = 1e-2

# maximal baseline uncertainty threshold.
# if the baseline uncertainty is larger than this value,
# re-bootstrappign will occur
MAX_BASELINE_UNCERTAINTY = .15

# minimal PnP / RANSAC inlier ratio from one frame to the next
MIN_INLIER_RATIO = .5

BA_DISTANCE_TH = 1e2

BUNDLE_ADJUSTMENT_KEYFRAME_LOOK_BACK = 20 if DATASET == 'PARKING' else 4