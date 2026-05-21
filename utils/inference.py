from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .anchor_utils import build_grid_centers
from .config import (
    CENTERNESS_POWER,
    CLASS_NAMES,
    CONF_THRESH,
    FEATURE_STRIDE,
    IMG_SIZE,
    MEAN,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    STD,
    VALID_IMAGE_EXTS,
)


def preprocess_image(image_path: Path, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, Dict[str, float]]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    meta = {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "sx": float(orig_w) / float(img_size),
        "sy": float(orig_h) / float(img_size),
    }
    return tensor, meta


def _decode_single(
    outputs: Dict[str, torch.Tensor],
    img_size: int,
    stride: float,
    conf_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_prob = torch.sigmoid(outputs["cls_logits"][0])
    reg = outputs["reg_preds"][0]
    center_prob = torch.sigmoid(outputs["center_logits"][0, 0])

    num_classes, h, w = cls_prob.shape
    device = cls_prob.device
    gx, gy = build_grid_centers(h, w, stride=stride, device=device)

    boxes_all = []
    scores_all = []
    cls_all = []

    for c in range(num_classes):
        score_map = cls_prob[c] * center_prob.pow(CENTERNESS_POWER)
        mask = score_map > conf_thresh
        if not mask.any():
            continue

        ys, xs = torch.where(mask)
        scores = score_map[ys, xs]

        l = reg[0, ys, xs] * stride
        t = reg[1, ys, xs] * stride
        r = reg[2, ys, xs] * stride
        b = reg[3, ys, xs] * stride

        cx = gx[ys, xs]
        cy = gy[ys, xs]

        x1 = (cx - l).clamp(0, img_size)
        y1 = (cy - t).clamp(0, img_size)
        x2 = (cx + r).clamp(0, img_size)
        y2 = (cy + b).clamp(0, img_size)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes_all.append(boxes.detach().cpu().numpy())
        scores_all.append(scores.detach().cpu().numpy())
        cls_all.append(np.full((boxes.shape[0],), c, dtype=np.int64))

    if not boxes_all:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(boxes_all, axis=0),
        np.concatenate(scores_all, axis=0),
        np.concatenate(cls_all, axis=0),
    )


def _remap_to_original(boxes: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] *= meta["sx"]
    boxes[:, [1, 3]] *= meta["sy"]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, meta["orig_w"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, meta["orig_h"])

    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou < iou_thresh]

    return np.asarray(keep, dtype=np.int64)


def apply_per_class_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    class_names: List[str],
    iou_thresh: float,
) -> List[Dict[str, Any]]:
    if len(boxes) == 0:
        return []

    results: List[Dict[str, Any]] = []
    for c in np.unique(cls_ids):
        mask = cls_ids == c
        b = boxes[mask]
        s = scores[mask]

        valid = np.isfinite(b).all(axis=1) & ((b[:, 2] - b[:, 0]) > 1.0) & ((b[:, 3] - b[:, 1]) > 1.0)
        b = b[valid]
        s = s[valid]
        if len(b) == 0:
            continue

        keep = _nms_numpy(b, s, iou_thresh=iou_thresh)
        for idx in keep:
            results.append(
                {
                    "class": class_names[int(c)],
                    "confidence": float(s[idx]),
                    "bbox": [float(v) for v in b[idx]],
                }
            )

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    img_size: int = IMG_SIZE,
    stride: float = FEATURE_STRIDE,
    conf_thresh: float = CONF_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
    class_names: List[str] = CLASS_NAMES,
) -> List[Dict[str, Any]]:
    image_t, meta = preprocess_image(image_path, img_size=img_size)
    outputs = model(image_t.to(device))

    boxes, scores, cls_ids = _decode_single(
        outputs=outputs,
        img_size=img_size,
        stride=float(stride),
        conf_thresh=conf_thresh,
    )
    boxes = _remap_to_original(boxes, meta)

    return apply_per_class_nms(
        boxes=boxes,
        scores=scores,
        cls_ids=cls_ids,
        class_names=class_names,
        iou_thresh=nms_iou_thresh,
    )


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    image_dir: str,
    device: torch.device,
    img_size: int = IMG_SIZE,
    stride: float = FEATURE_STRIDE,
    conf_thresh: float = CONF_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
    class_names: List[str] = CLASS_NAMES,
) -> List[Dict[str, Any]]:
    folder = Path(image_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])
    results: List[Dict[str, Any]] = []

    model.eval()
    for path in paths:
        image_id = os.path.basename(path)
        try:
            boxes = predict_single_image(
                model=model,
                image_path=path,
                device=device,
                img_size=img_size,
                stride=stride,
                conf_thresh=conf_thresh,
                nms_iou_thresh=nms_iou_thresh,
                class_names=class_names,
            )
            results.append({"image_id": image_id, "boxes": boxes})
        except Exception as exc:
            print(f"[WARN] Failed {image_id}: {exc}")
            results.append({"image_id": image_id, "boxes": []})

    return results
