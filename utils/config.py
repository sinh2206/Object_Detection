from __future__ import annotations

NUM_CLASSES = 5
CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_LOSS_WEIGHTS = [1.0, 2.5, 3.0, 3.0, 2.0]

# Input and feature geometry
IMG_SIZE = 448
GRID_SIZE = 14
FEATURE_STRIDE = IMG_SIZE // GRID_SIZE  # 32 for default config
STRIDES = [FEATURE_STRIDE]

# Kept for compatibility with previous code style
NUM_ANCHORS = 1
FPN_CHANNELS = 256
ANCHOR_SIZES = {FEATURE_STRIDE: [(32, 32)]}

# Loss parameters
IOU_POS_THRESH = 0.50
IOU_IGNORE_THRESH = 0.40
LAMBDA_OBJ = 1.0
LAMBDA_BOX = 1.0
LAMBDA_CLS = 2.0
LAMBDA_CENTER = 1.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.12
NMS_IOU_THRESH = 0.50
CENTERNESS_POWER = 1.0

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = 1e-6
