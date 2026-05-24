from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 320

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
PERSON_CLASS_NAME = "person"

STRIDES = [16, 32]
FPN_CHANNELS = 128

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.55
NMS_IOU_THRESH = 0.3
AGNOSTIC_NMS_IOU_THRESH = 0.75
CROSS_CLASS_IOU_THRESH = 0.85
CROSS_CLASS_CONTAIN_THRESH = 0.9
SAME_CLASS_CONTAIN_THRESH = 0.88

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LAMBDA_CLS = 1.0
LAMBDA_REG = 1.0
LAMBDA_CTR = 0.5
LABEL_SMOOTHING = 0.05
