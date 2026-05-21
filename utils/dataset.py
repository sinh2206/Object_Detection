from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .anchor_utils import compute_centerness
from .config import CLASS_TO_IDX, GRID_SIZE, IMG_SIZE, NUM_CLASSES


class DetectionDataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        img_dir: str,
        transforms: Optional[Callable] = None,
        img_size: int = IMG_SIZE,
        grid_size: int = GRID_SIZE,
        center_sampling_radius: float = 1.5,
    ):
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.transforms = transforms
        self.img_size = int(img_size)
        self.grid_size = int(grid_size)
        self.stride = float(self.img_size / self.grid_size)
        self.center_sampling_radius = float(center_sampling_radius)

        self.img_info = {img["id"]: img for img in data["images"]}
        self.ann_map = defaultdict(list)
        for ann in data["annotations"]:
            self.ann_map[ann["image_id"]].append(ann)

        self.image_ids = list(self.img_info.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_targets(self, bboxes: List[List[float]], labels: List[int]) -> Dict[str, torch.Tensor]:
        g = self.grid_size
        cls_target = np.zeros((NUM_CLASSES, g, g), dtype=np.float32)
        reg_target = np.zeros((4, g, g), dtype=np.float32)
        center_target = np.zeros((1, g, g), dtype=np.float32)
        pos_mask = np.zeros((1, g, g), dtype=np.float32)

        area_map = np.full((g, g), np.inf, dtype=np.float32)
        center_radius_px = self.center_sampling_radius * self.stride

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = (x2 - x1) * (y2 - y1)
            if area <= 0.0:
                continue

            gx_min = max(int(np.floor(x1 / self.stride)), 0)
            gy_min = max(int(np.floor(y1 / self.stride)), 0)
            gx_max = min(int(np.floor((x2 - 1e-6) / self.stride)), g - 1)
            gy_max = min(int(np.floor((y2 - 1e-6) / self.stride)), g - 1)
            if gx_max < gx_min or gy_max < gy_min:
                continue

            cx_min = cx - center_radius_px
            cy_min = cy - center_radius_px
            cx_max = cx + center_radius_px
            cy_max = cy + center_radius_px

            for gy in range(gy_min, gy_max + 1):
                cell_cy = (gy + 0.5) * self.stride
                if cell_cy < cy_min or cell_cy > cy_max:
                    continue

                for gx in range(gx_min, gx_max + 1):
                    cell_cx = (gx + 0.5) * self.stride
                    if cell_cx < cx_min or cell_cx > cx_max:
                        continue

                    l = cell_cx - x1
                    t = cell_cy - y1
                    r = x2 - cell_cx
                    b = y2 - cell_cy
                    if min(l, t, r, b) <= 0.0:
                        continue

                    if area >= area_map[gy, gx]:
                        continue

                    area_map[gy, gx] = area
                    cls_target[:, gy, gx] = 0.0
                    cls_target[label, gy, gx] = 1.0
                    # Normalize ltrb by stride to stabilize regression optimization.
                    reg_target[:, gy, gx] = np.array([l, t, r, b], dtype=np.float32) / float(self.stride)
                    center_target[0, gy, gx] = compute_centerness(l, t, r, b)
                    pos_mask[0, gy, gx] = 1.0

        return {
            "cls": torch.from_numpy(cls_target),
            "reg": torch.from_numpy(reg_target),
            "center": torch.from_numpy(center_target),
            "pos_mask": torch.from_numpy(pos_mask),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        image_id = self.image_ids[idx]
        info = self.img_info[image_id]
        image_path = os.path.join(self.img_dir, os.path.basename(info["file_name"]))

        image = cv2.imread(image_path)
        if image is None:
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        bboxes: List[List[float]] = []
        labels: List[int] = []
        for ann in self.ann_map.get(image_id, []):
            if ann.get("class") not in CLASS_TO_IDX:
                continue
            x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
            x1 = max(0.0, min(x1, float(w - 1)))
            y1 = max(0.0, min(y1, float(h - 1)))
            x2 = max(x1 + 1.0, min(x2, float(w)))
            y2 = max(y1 + 1.0, min(y2, float(h)))
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[ann["class"]])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=labels)
            image_tensor = transformed["image"]
            bboxes = [list(map(float, b)) for b in transformed["bboxes"]]
            labels = [int(v) for v in transformed["class_labels"]]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        valid_boxes: List[List[float]] = []
        valid_labels: List[int] = []
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(x1, float(self.img_size - 1)))
            y1 = max(0.0, min(y1, float(self.img_size - 1)))
            x2 = max(x1 + 1.0, min(x2, float(self.img_size)))
            y2 = max(y1 + 1.0, min(y2, float(self.img_size)))
            if x2 > x1 and y2 > y1:
                valid_boxes.append([x1, y1, x2, y2])
                valid_labels.append(label)

        targets = self._build_targets(valid_boxes, valid_labels)
        return image_tensor, targets, image_id


def collate_fn(batch):
    images, targets, image_ids = zip(*batch)
    images = torch.stack(images, dim=0)

    stacked_targets = {
        "cls": torch.stack([t["cls"] for t in targets], dim=0),
        "reg": torch.stack([t["reg"] for t in targets], dim=0),
        "center": torch.stack([t["center"] for t in targets], dim=0),
        "pos_mask": torch.stack([t["pos_mask"] for t in targets], dim=0),
    }

    return images, stacked_targets, list(image_ids)
