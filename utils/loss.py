from __future__ import annotations

from typing import Dict, Tuple
from typing import Optional

import torch
import torch.nn as nn

from .config import CLASS_LOSS_WEIGHTS, FOCAL_ALPHA, FOCAL_GAMMA, LAMBDA_BOX, LAMBDA_CENTER, LAMBDA_CLS

_bce = nn.BCEWithLogitsLoss(reduction="mean")
_smooth_l1 = nn.SmoothL1Loss(reduction="mean")


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    class_weights: Optional[torch.Tensor] = None,
):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1, 1, 1)
    return loss.mean()


def compute_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
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

        reg_mask = pos_mask.expand(-1, 4, -1, -1)
        loss_reg = _smooth_l1(reg_preds[reg_mask], reg_target[reg_mask])
    else:
        loss_center = cls_logits.new_zeros(())
        loss_reg = cls_logits.new_zeros(())

    total_loss = LAMBDA_CLS * loss_cls + LAMBDA_CENTER * loss_center + LAMBDA_BOX * loss_reg

    parts = {
        "cls": float(loss_cls.item()),
        "center": float(loss_center.item()),
        "reg": float(loss_reg.item()),
    }
    return total_loss, parts
