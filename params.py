# General
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
KLT_N_ITERS: int = 50
KLT_NUM_PYRAMIDS = 8
