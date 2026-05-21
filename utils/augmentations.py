from __future__ import annotations

import inspect

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .config import MEAN, STD


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.2,
        clip=True,
    )


def _supported_kwargs(transform_cls, **kwargs):
    params = inspect.signature(transform_cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _pad_to_square(img_size: int) -> A.BasicTransform:
    return A.PadIfNeeded(
        min_height=img_size,
        min_width=img_size,
        **_supported_kwargs(
            A.PadIfNeeded,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            fill=(114, 114, 114),
        ),
    )


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            _pad_to_square(img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.20,
                rotate_limit=12,
                **_supported_kwargs(
                    A.ShiftScaleRotate,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(114, 114, 114),
                    fill=(114, 114, 114),
                ),
                p=0.45,
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.10, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.25, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30, val_shift_limit=18, p=1.0),
                ],
                p=0.75,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(
                        **_supported_kwargs(
                            A.GaussNoise,
                            var_limit=(8.0, 40.0),
                            std_range=(0.02, 0.08),
                            mean_range=(0.0, 0.0),
                        ),
                        p=1.0,
                    ),
                ],
                p=0.25,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            _pad_to_square(img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )
