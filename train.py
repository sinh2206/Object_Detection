from __future__ import annotations

"""
Anchor-Free training entrypoint (FCOS/YOLOX-style simplified).

1) Data and augmentation
- Read train/val annotations.
- Apply Albumentations transforms.
- Dataset builds per-image grid targets: cls, centerness, l/t/r/b.

2) Model
- ResNet18/34 backbone without avgpool/fc.
- Detection head outputs cls logits, positive box distances, center logits.

3) Loss
- Classification BCEWithLogits on all grid cells.
- Centerness BCEWithLogits on positive cells only.
- Regression SmoothL1 on positive cells only.

4) Training loop
- AdamW + Cosine LR scheduler.
- Periodic validation (and optional mAP@0.5).
- Save best checkpoint to ./models/best.pth.
"""

import argparse
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.augmentations import get_train_transforms, get_val_transforms
from utils.config import (
    CLASS_NAMES,
    CONF_THRESH,
    FEATURE_STRIDE,
    GRID_SIZE,
    IMG_SIZE,
    NMS_IOU_THRESH,
)
from utils.dataset import DetectionDataset, collate_fn
from utils.loss import compute_loss
from utils.metrics import evaluate_map
from utils.model import AnchorFreeDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train anchor-free detector")
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--val_image_dir", required=True)
    p.add_argument("--checkpoint_dir", default="./models")

    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--grid_size", type=int, default=GRID_SIZE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")

    p.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    p.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--compute_map", action="store_true")

    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in targets.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
):
    model.train()
    total = 0.0
    n = 0
    parts_sum = {"cls": 0.0, "center": 0.0, "reg": 0.0}

    for images, targets, _ in loader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, parts = compute_loss(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, parts = compute_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total += float(loss.item())
        for k in parts_sum:
            parts_sum[k] += float(parts[k])
        n += 1

    avg_loss = total / max(n, 1)
    avg_parts = {k: (v / max(n, 1)) for k, v in parts_sum.items()}
    return avg_loss, avg_parts


@torch.no_grad()
def validate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0.0
    n = 0
    parts_sum = {"cls": 0.0, "center": 0.0, "reg": 0.0}

    for images, targets, _ in loader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images)
        loss, parts = compute_loss(outputs, targets)

        total += float(loss.item())
        for k in parts_sum:
            parts_sum[k] += float(parts[k])
        n += 1

    avg_loss = total / max(n, 1)
    avg_parts = {k: (v / max(n, 1)) for k, v in parts_sum.items()}
    return avg_loss, avg_parts


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    best_map: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_map": best_map,
            "img_size": args.img_size,
            "grid_size": args.grid_size,
            "stride": float(args.img_size / args.grid_size),
            "class_names": CLASS_NAMES,
            "backbone": args.backbone,
            "model_type": "anchor_free_fcos_simplified",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.img_size % args.grid_size != 0:
        raise ValueError("img_size must be divisible by grid_size.")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DetectionDataset(
        ann_path=args.train_data,
        img_dir=args.image_dir,
        transforms=get_train_transforms(args.img_size),
        img_size=args.img_size,
        grid_size=args.grid_size,
    )
    val_ds = DetectionDataset(
        ann_path=args.val_data,
        img_dir=args.val_image_dir,
        transforms=get_val_transforms(args.img_size),
        img_size=args.img_size,
        grid_size=args.grid_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = AnchorFreeDetector(
        num_classes=len(CLASS_NAMES),
        backbone_name=args.backbone,
        pretrained=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 1
    best_val_loss = float("inf")
    best_map = 0.0
    best_score = -float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_map = float(ckpt.get("best_map", best_map))
        else:
            model.load_state_dict(ckpt, strict=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_parts = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va_loss, va_parts = validate_loss(model, val_loader, device)
        scheduler.step()

        score = -va_loss
        map50 = None
        if args.compute_map and (epoch % max(1, args.eval_every) == 0):
            map50 = evaluate_map(
                model=model,
                val_ann_path=args.val_data,
                val_img_dir=args.val_image_dir,
                device=device,
                img_size=args.img_size,
                stride=float(args.img_size / args.grid_size),
                conf_thresh=args.conf_thresh,
                nms_iou_thresh=args.nms_thresh,
            )
            best_map = max(best_map, map50)
            score = map50

        if va_loss < best_val_loss:
            best_val_loss = va_loss

        if score > best_score:
            best_score = score
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            save_checkpoint(best_path, model, optimizer, scheduler, args, epoch, best_val_loss, best_map)

        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_e{epoch:03d}.pth")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, args, epoch, best_val_loss, best_map)

        msg = (
            f"[Epoch {epoch:03d}] "
            f"train={tr_loss:.4f} (cls={tr_parts['cls']:.3f}, ctr={tr_parts['center']:.3f}, reg={tr_parts['reg']:.3f}) "
            f"val={va_loss:.4f} (cls={va_parts['cls']:.3f}, ctr={va_parts['center']:.3f}, reg={va_parts['reg']:.3f})"
        )
        if map50 is not None:
            msg += f" mAP@0.5={map50:.4f}"
        msg += f" time={time.time() - t0:.1f}s"
        print(msg)

    best_path = os.path.join(args.checkpoint_dir, "best.pth")
    if not os.path.isfile(best_path):
        save_checkpoint(best_path, model, optimizer, scheduler, args, args.epochs, best_val_loss, best_map)


if __name__ == "__main__":
    main()
