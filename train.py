from __future__ import annotations

"""
Anchor-Free training entrypoint (FCOS/YOLOX-style simplified).

Enhancements in this version:
- FCOS-style target assignment on grid centers inside each bbox.
- Stronger image preprocessing/augmentation.
- EMA model weights for stabler validation and checkpoint selection.
- Separate LR for backbone/head + warmup + cosine schedule.
"""

import argparse
import copy
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.augmentations import get_train_transforms, get_val_transforms
from utils.config import CLASS_NAMES, CONF_THRESH, GRID_SIZE, IMG_SIZE, NMS_IOU_THRESH
from utils.dataset import DetectionDataset, collate_fn
from utils.loss import compute_loss
from utils.metrics import evaluate_map
from utils.model import AnchorFreeDetector


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()
            if not ema_v.dtype.is_floating_point:
                ema_v.copy_(model_v)
                continue
            ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


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
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet34")

    p.add_argument("--backbone_lr_mult", type=float, default=0.35)
    p.add_argument("--warmup_epochs", type=int, default=4)
    p.add_argument("--ema_decay", type=float, default=0.9995)

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


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ]
    return optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace) -> optim.lr_scheduler.LRScheduler:
    warmup_epochs = min(max(0, args.warmup_epochs), max(0, args.epochs - 1))
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=args.lr * 0.02,
        )
        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.02,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    stride: float,
    ema: Optional[ModelEMA] = None,
) -> Tuple[float, Dict[str, float]]:
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
                loss, parts = compute_loss(outputs, targets, stride=stride)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, parts = compute_loss(outputs, targets, stride=stride)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total += float(loss.item())
        for k in parts_sum:
            parts_sum[k] += float(parts[k])
        n += 1

    avg_loss = total / max(n, 1)
    avg_parts = {k: (v / max(n, 1)) for k, v in parts_sum.items()}
    return avg_loss, avg_parts


@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    stride: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total = 0.0
    n = 0
    parts_sum = {"cls": 0.0, "center": 0.0, "reg": 0.0}

    for images, targets, _ in loader:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images)
        loss, parts = compute_loss(outputs, targets, stride=stride)

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
    scheduler: optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    best_map: float,
    ema_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    payload = {
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
    }
    if ema_state_dict is not None:
        payload["ema_state_dict"] = ema_state_dict
    torch.save(payload, path)


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

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    model = AnchorFreeDetector(
        num_classes=len(CLASS_NAMES),
        backbone_name=args.backbone,
        pretrained=True,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

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
            if ema is not None and "ema_state_dict" in ckpt:
                ema.ema.load_state_dict(ckpt["ema_state_dict"], strict=True)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_map = float(ckpt.get("best_map", best_map))
        else:
            model.load_state_dict(ckpt, strict=True)
            if ema is not None:
                ema.ema.load_state_dict(ckpt, strict=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        stride = float(args.img_size / args.grid_size)
        tr_loss, tr_parts = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            stride=stride,
            ema=ema,
        )

        eval_model = ema.ema if ema is not None else model
        va_loss, va_parts = validate_loss(eval_model, val_loader, device, stride=stride)
        scheduler.step()

        score = -va_loss
        map50 = None
        if args.compute_map and (epoch % max(1, args.eval_every) == 0):
            map50 = evaluate_map(
                model=eval_model,
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
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_map=best_map,
                ema_state_dict=(ema.ema.state_dict() if ema is not None else None),
            )

        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_e{epoch:03d}.pth")
        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_map=best_map,
            ema_state_dict=(ema.ema.state_dict() if ema is not None else None),
        )

        lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups])
        msg = (
            f"[Epoch {epoch:03d}] "
            f"lr=[{lrs}] "
            f"train={tr_loss:.4f} (cls={tr_parts['cls']:.3f}, ctr={tr_parts['center']:.3f}, reg={tr_parts['reg']:.3f}) "
            f"val={va_loss:.4f} (cls={va_parts['cls']:.3f}, ctr={va_parts['center']:.3f}, reg={va_parts['reg']:.3f})"
        )
        if map50 is not None:
            msg += f" mAP@0.5={map50:.4f}"
        msg += f" time={time.time() - t0:.1f}s"
        print(msg)

    best_path = os.path.join(args.checkpoint_dir, "best.pth")
    if not os.path.isfile(best_path):
        save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=args.epochs,
            best_val_loss=best_val_loss,
            best_map=best_map,
            ema_state_dict=(ema.ema.state_dict() if ema is not None else None),
        )


if __name__ == "__main__":
    main()
