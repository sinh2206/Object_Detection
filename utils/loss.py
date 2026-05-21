from __future__ import annotations

from typing import Dict, Tuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASS_LOSS_WEIGHTS, FOCAL_ALPHA, FOCAL_GAMMA, LAMBDA_BOX, LAMBDA_CENTER, LAMBDA_CLS

_bce = nn.BCEWithLogitsLoss(reduction="mean")


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


def _ltrb_iou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    pred = pred_ltrb.clamp(min=0.0)
    target = target_ltrb.clamp(min=0.0)

    inter_w = torch.minimum(pred[:, 0], target[:, 0]) + torch.minimum(pred[:, 2], target[:, 2])
    inter_h = torch.minimum(pred[:, 1], target[:, 1]) + torch.minimum(pred[:, 3], target[:, 3])
    inter = inter_w.clamp(min=0.0) * inter_h.clamp(min=0.0)

    area_pred = (pred[:, 0] + pred[:, 2]).clamp(min=0.0) * (pred[:, 1] + pred[:, 3]).clamp(min=0.0)
    area_tgt = (target[:, 0] + target[:, 2]).clamp(min=0.0) * (target[:, 1] + target[:, 3]).clamp(min=0.0)
    union = area_pred + area_tgt - inter + 1e-6
    iou = inter / union
    return 1.0 - iou


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
        pos_flat = pos_mask[:, 0]

        pred_reg = reg_preds.permute(0, 2, 3, 1)[pos_flat]
        tgt_reg = reg_target.permute(0, 2, 3, 1)[pos_flat]
        tgt_ctr = center_target[:, 0][pos_flat].detach().clamp(min=0.05)

        reg_l1 = F.smooth_l1_loss(pred_reg, tgt_reg, reduction="none").mean(dim=1)
        reg_iou = _ltrb_iou_loss(pred_reg, tgt_reg)
        reg_mix = reg_l1 + reg_iou
        loss_reg = (reg_mix * tgt_ctr).sum() / tgt_ctr.sum().clamp(min=1e-6)
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
