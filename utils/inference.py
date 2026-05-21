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
    STD,
    VALID_IMAGE_EXTS,
)


def _letterbox_image(
    image: Image.Image,
    img_size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, Dict[str, float]]:
    orig_w, orig_h = image.size
    scale = min(float(img_size) / float(orig_w), float(img_size) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), color=fill)
    pad_x = int((img_size - new_w) // 2)
    pad_y = int((img_size - new_h) // 2)
    canvas.paste(resized, (pad_x, pad_y))

    meta = {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "new_w": float(new_w),
        "new_h": float(new_h),
    }
    return canvas, meta


def preprocess_image(image_path: Path, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, Dict[str, float]]:
    image = Image.open(image_path).convert("RGB")
    canvas, meta = _letterbox_image(image=image, img_size=img_size)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, meta


def _decode_single(
    outputs: Dict[str, torch.Tensor],
    img_size: int,
    stride: float,
    conf_thresh: float,
    topk_per_class: int = 500,
    min_box_size: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_prob = torch.sigmoid(outputs["cls_logits"][0])
    reg = outputs["reg_preds"][0]
    center_prob = torch.sigmoid(outputs["center_logits"][0, 0])

    num_classes, h, w = cls_prob.shape
    device = cls_prob.device
    gx, gy = build_grid_centers(h, w, stride=stride, device=device)

    boxes_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    cls_all: List[np.ndarray] = []

    for c in range(num_classes):
        score_map = cls_prob[c] * center_prob.pow(CENTERNESS_POWER)
        mask = score_map > conf_thresh
        if not mask.any():
            continue

        ys, xs = torch.where(mask)
        scores = score_map[ys, xs]
        if topk_per_class > 0 and scores.numel() > topk_per_class:
            scores, topk_idx = torch.topk(scores, k=topk_per_class, dim=0, largest=True, sorted=False)
            ys = ys[topk_idx]
            xs = xs[topk_idx]

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

        keep = ((x2 - x1) >= min_box_size) & ((y2 - y1) >= min_box_size)
        if not keep.any():
            continue

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)[keep]
        scores = scores[keep]

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
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_x"]) / max(meta["scale"], 1e-6)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_y"]) / max(meta["scale"], 1e-6)

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

    keep: List[int] = []
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
    max_detections: int = 100,
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
    if max_detections > 0:
        results = results[:max_detections]
    return results


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_path: Path | str,
    device: torch.device,
    img_size: int = IMG_SIZE,
    stride: float = FEATURE_STRIDE,
    conf_thresh: float = CONF_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
    class_names: List[str] = CLASS_NAMES,
    topk_per_class: int = 500,
    max_detections: int = 100,
    min_box_size: float = 2.0,
    tta_flip: bool = True,
) -> List[Dict[str, Any]]:
    image_path = Path(image_path)
    image_t, meta = preprocess_image(image_path, img_size=img_size)

    outputs = model(image_t.to(device))
    boxes, scores, cls_ids = _decode_single(
        outputs=outputs,
        img_size=img_size,
        stride=float(stride),
        conf_thresh=conf_thresh,
        topk_per_class=topk_per_class,
        min_box_size=min_box_size,
    )

    if tta_flip:
        flip_t = torch.flip(image_t, dims=[3])
        outputs_f = model(flip_t.to(device))
        b2, s2, c2 = _decode_single(
            outputs=outputs_f,
            img_size=img_size,
            stride=float(stride),
            conf_thresh=conf_thresh,
            topk_per_class=topk_per_class,
            min_box_size=min_box_size,
        )
        if len(b2) > 0:
            b2 = b2.copy()
            x1 = b2[:, 0].copy()
            x2 = b2[:, 2].copy()
            b2[:, 0] = float(img_size) - x2
            b2[:, 2] = float(img_size) - x1

        if len(b2) > 0 and len(boxes) > 0:
            boxes = np.concatenate([boxes, b2], axis=0)
            scores = np.concatenate([scores, s2], axis=0)
            cls_ids = np.concatenate([cls_ids, c2], axis=0)
        elif len(b2) > 0:
            boxes, scores, cls_ids = b2, s2, c2

    boxes = _remap_to_original(boxes, meta)
    return apply_per_class_nms(
        boxes=boxes,
        scores=scores,
        cls_ids=cls_ids,
        class_names=class_names,
        iou_thresh=nms_iou_thresh,
        max_detections=max_detections,
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
    topk_per_class: int = 500,
    max_detections: int = 100,
    min_box_size: float = 2.0,
    tta_flip: bool = True,
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
                topk_per_class=topk_per_class,
                max_detections=max_detections,
                min_box_size=min_box_size,
                tta_flip=tta_flip,
            )
            results.append({"image_id": image_id, "boxes": boxes})
        except Exception as exc:
            print(f"[WARN] Failed {image_id}: {exc}")
            results.append({"image_id": image_id, "boxes": []})

    return results
