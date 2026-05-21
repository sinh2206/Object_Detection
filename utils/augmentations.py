from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import MEAN, STD


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.2,
        clip=True,
    )


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=0.6),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )
