from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import CLASS_TO_IDX, CLASS_NAMES, CONF_THRESH, FEATURE_STRIDE, IMG_SIZE, NMS_IOU_THRESH
from .inference import predict_single_image


def _bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_b = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_a = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area_b + area_a - inter + 1e-6)


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    val_ann_path: str,
    val_img_dir: str,
    device: torch.device,
    img_size: int = IMG_SIZE,
    stride: float = FEATURE_STRIDE,
    conf_thresh: float = CONF_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
) -> float:
    with open(val_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map: Dict[str, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann["class"] not in CLASS_TO_IDX:
            continue
        c = CLASS_TO_IDX[ann["class"]]
        x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
        gt_map[ann["image_id"]].append((c, x1, y1, x2, y2))

    img_info = {img["id"]: img for img in data["images"]}

    all_pred = []
    for image_id, meta in img_info.items():
        image_path = os.path.join(val_img_dir, os.path.basename(meta["file_name"]))
        dets = predict_single_image(
            model=model,
            image_path=image_path,
            device=device,
            img_size=img_size,
            stride=stride,
            conf_thresh=conf_thresh,
            nms_iou_thresh=nms_iou_thresh,
            class_names=CLASS_NAMES,
        )
        for d in dets:
            c = CLASS_TO_IDX[d["class"]]
            all_pred.append((image_id, c, float(d["confidence"]), d["bbox"]))

    ap_list = []
    num_classes = len(CLASS_NAMES)

    for c in range(num_classes):
        preds_c = [(iid, conf, bbox) for iid, cc, conf, bbox in all_pred if cc == c]
        preds_c.sort(key=lambda x: -x[1])

        n_gt = sum(1 for iid in gt_map for g in gt_map[iid] if g[0] == c)
        if n_gt == 0:
            continue

        tp = np.zeros(len(preds_c), dtype=np.float32)
        fp = np.zeros(len(preds_c), dtype=np.float32)
        matched = defaultdict(set)

        for i, (iid, _, bbox) in enumerate(preds_c):
            gt_boxes = [g[1:] for g in gt_map.get(iid, []) if g[0] == c]
            if not gt_boxes:
                fp[i] = 1.0
                continue

            gt_arr = np.asarray(gt_boxes, dtype=np.float32)
            ious = _bbox_iou(np.asarray(bbox, dtype=np.float32), gt_arr)
            best_j = int(np.argmax(ious))
            best_iou = float(ious[best_j])

            if best_iou >= 0.5 and best_j not in matched[iid]:
                tp[i] = 1.0
                matched[iid].add(best_j)
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (n_gt + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)

        ap = 0.0
        for thr in np.linspace(0.0, 1.0, 11):
            p = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
            ap += p / 11.0

        ap_list.append(ap)

    return float(np.mean(ap_list)) if ap_list else 0.0
