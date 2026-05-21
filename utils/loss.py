from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .anchor_utils import giou_loss, ltrb_to_xyxy
from .config import (
    CLASS_LOSS_WEIGHTS,
    FEATURE_STRIDE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LAMBDA_BOX,
    LAMBDA_CENTER,
    LAMBDA_CLS,
)

_bce = nn.BCEWithLogitsLoss(reduction="mean")


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1, 1, 1)
    return loss.mean()


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    stride: float = FEATURE_STRIDE,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls_logits = outputs["cls_logits"]
    reg_preds = outputs["reg_preds"]
    center_logits = outputs["center_logits"]

    cls_target = targets["cls"]
    reg_target = targets["reg"]
    center_target = targets["center"]
    pos_mask = targets["pos_mask"] > 0.5

    cls_weights = cls_logits.new_tensor(CLASS_LOSS_WEIGHTS, dtype=cls_logits.dtype)
    loss_cls = focal_bce_loss(cls_logits, cls_target, class_weights=cls_weights)

    if pos_mask.any():
        loss_center = _bce(center_logits[pos_mask], center_target[pos_mask])

        pos_flat = pos_mask[:, 0]
        pred_reg = reg_preds.permute(0, 2, 3, 1)[pos_flat]
        tgt_reg = reg_target.permute(0, 2, 3, 1)[pos_flat]

        _, ys, xs = torch.where(pos_flat)
        cx = (xs.to(pred_reg.dtype) + 0.5) * float(stride)
        cy = (ys.to(pred_reg.dtype) + 0.5) * float(stride)

        pred_xyxy = ltrb_to_xyxy(cx, cy, pred_reg)
        tgt_xyxy = ltrb_to_xyxy(cx, cy, tgt_reg)
        loss_reg = giou_loss(pred_xyxy, tgt_xyxy).mean()
    else:
        loss_center = cls_logits.new_zeros(())
        loss_reg = cls_logits.new_zeros(())

    total_loss = LAMBDA_CLS * loss_cls + LAMBDA_BOX * loss_reg + LAMBDA_CENTER * loss_center
    parts = {
        "cls": float(loss_cls.item()),
        "reg": float(loss_reg.item()),
        "center": float(loss_center.item()),
    }
    return total_loss, parts
