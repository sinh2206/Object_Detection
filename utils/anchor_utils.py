from __future__ import annotations

from typing import Tuple

import torch

from .config import EPS


def build_grid_centers(feat_h: int, feat_w: int, stride: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def compute_centerness(l: float, t: float, r: float, b: float) -> float:
    lr_min, lr_max = min(l, r), max(l, r)
    tb_min, tb_max = min(t, b), max(t, b)
    if lr_min <= 0.0 or tb_min <= 0.0:
        return 0.0
    return float(((lr_min / (lr_max + EPS)) * (tb_min / (tb_max + EPS))) ** 0.5)


def ltrb_to_xyxy(cx: torch.Tensor, cy: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    x1 = cx - ltrb[..., 0]
    y1 = cy - ltrb[..., 1]
    x2 = cx + ltrb[..., 2]
    y2 = cy + ltrb[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iou_pairwise(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]

    a = boxes_a.unsqueeze(1).expand(n, m, 4)
    b = boxes_b.unsqueeze(0).expand(n, m, 4)

    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter + EPS
    return inter / union


def giou_loss(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = pred_xyxy.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(dim=-1)

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + EPS
    iou = inter / union

    cx1 = torch.minimum(px1, gx1)
    cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2)
    cy2 = torch.maximum(py2, gy2)
    area_c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + EPS
    giou = iou - (area_c - union) / area_c
    return 1.0 - giou
