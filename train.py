"""
train.py  –  Object Detection Training Script (Anchor-Free / FCOS-style)

Usage:
    python train.py \
        --train_data  ./public/annotations/train.json \
        --val_data    ./public/annotations/val.json \
        --image_dir   ./public/train/images \
        --val_image_dir ./public/val/images \
        --checkpoint_dir ./models/

The script will save the best checkpoint (by val mAP@0.5) to
<checkpoint_dir>/best.pth and a periodic checkpoint to
<checkpoint_dir>/last.pth.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLASS_NAMES,
    CONF_THRESH,
    FEATURE_STRIDE,
    GRID_SIZE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    NUM_CLASSES,
)
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.dataset import DetectionDataset, collate_fn
from utils.loss import compute_loss
from utils.metrics import evaluate_map
from utils.model import AnchorFreeDetector


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train anchor-free object detector")
    p.add_argument("--train_data",     required=True,  help="Path to train annotation JSON")
    p.add_argument("--val_data",       required=True,  help="Path to val annotation JSON")
    p.add_argument("--image_dir",      required=True,  help="Path to train images directory")
    p.add_argument("--val_image_dir",  required=True,  help="Path to val images directory")
    p.add_argument("--checkpoint_dir", default="./models", help="Where to save checkpoints")
    # training hyper-params
    p.add_argument("--epochs",      type=int,   default=40,   help="Total training epochs")
    p.add_argument("--batch_size",  type=int,   default=16,   help="Batch size")
    p.add_argument("--img_size",    type=int,   default=IMG_SIZE, help="Input image size")
    p.add_argument("--lr",          type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--workers",     type=int,   default=4,    help="DataLoader workers")
    p.add_argument("--backbone",    default="resnet34",
                   choices=["resnet34", "resnet18"], help="CNN backbone")
    p.add_argument("--resume",      default=None, help="Checkpoint path to resume from")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not use ImageNet pretrained backbone")
    p.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    p.add_argument("--nms_thresh",  type=float, default=NMS_IOU_THRESH)
    return p.parse_args()


def build_datasets(args) -> Tuple[DataLoader, DataLoader]:
    train_tfm = get_train_transforms(args.img_size)
    val_tfm   = get_val_transforms(args.img_size)

    grid = args.img_size // FEATURE_STRIDE

    train_ds = DetectionDataset(
        ann_path=args.train_data,
        img_dir=args.image_dir,
        transforms=train_tfm,
        img_size=args.img_size,
        grid_size=grid,
    )
    val_ds = DetectionDataset(
        ann_path=args.val_data,
        img_dir=args.val_image_dir,
        transforms=val_tfm,
        img_size=args.img_size,
        grid_size=grid,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: AnchorFreeDetector,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler,
) -> Dict[str, float]:
    model.train()
    total_loss = cls_loss = reg_loss = cen_loss = 0.0
    n = 0

    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss, parts = compute_loss(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        total_loss += loss.item()
        cls_loss   += parts["cls"]
        reg_loss   += parts["reg"]
        cen_loss   += parts["center"]
        n += 1

        if (step + 1) % 10 == 0:
            print(
                f"  [epoch {epoch}  step {step+1}/{len(loader)}]"
                f"  loss={total_loss/n:.4f}"
                f"  cls={cls_loss/n:.4f}"
                f"  reg={reg_loss/n:.4f}"
                f"  cen={cen_loss/n:.4f}"
            )

    return {
        "loss": total_loss / max(n, 1),
        "cls":  cls_loss  / max(n, 1),
        "reg":  reg_loss  / max(n, 1),
        "center": cen_loss / max(n, 1),
    }


@torch.no_grad()
def val_loss_epoch(
    model: AnchorFreeDetector,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average validation loss (fast proxy before full mAP)."""
    model.eval()
    total = 0.0
    n = 0
    for images, targets, _ in loader:
        images  = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        outputs = model(images)
        loss, _ = compute_loss(outputs, targets)
        total  += loss.item()
        n += 1
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_loader, val_loader = build_datasets(args)
    print(f"  Train: {len(train_loader.dataset)} images  |  Val: {len(val_loader.dataset)} images")

    # ── Model ────────────────────────────────────────────────────────────────
    pretrained = not args.no_pretrained
    model = AnchorFreeDetector(
        num_classes=NUM_CLASSES,
        backbone_name=args.backbone,
        pretrained=pretrained,
    ).to(device)
    print(f"Model: AnchorFreeDetector  backbone={args.backbone}  pretrained={pretrained}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    # Use lower LR for frozen backbone params, higher for head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "layer" in name or "stem" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warm-up (first 5 epochs warm up linearly)
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision scaler (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_map    = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map    = ckpt.get("best_map", 0.0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, best_map={best_map:.4f})")

    # ── Training Loop ────────────────────────────────────────────────────────
    MAP_EVAL_EVERY = 5   # compute full mAP every N epochs (expensive)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  |  LR={scheduler.get_last_lr()}")

        # -- Train
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)

        # -- Val loss (every epoch, cheap)
        v_loss = val_loss_epoch(model, val_loader, device)
        print(
            f"  → train_loss={train_stats['loss']:.4f}  val_loss={v_loss:.4f}"
            f"  time={time.time()-t0:.1f}s"
        )

        scheduler.step()

        # -- Save last checkpoint
        last_ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map":  best_map,
        }
        torch.save(last_ckpt, ckpt_dir / "last.pth")

        # -- Full mAP evaluation (every MAP_EVAL_EVERY epochs or last epoch)
        if epoch % MAP_EVAL_EVERY == 0 or epoch == args.epochs:
            print("  Computing mAP@0.5 on val set …")
            model.eval()
            val_map = evaluate_map(
                model=model,
                val_ann_path=args.val_data,
                val_img_dir=args.val_image_dir,
                device=device,
                img_size=args.img_size,
                stride=float(FEATURE_STRIDE),
                conf_thresh=args.conf_thresh,
                nms_iou_thresh=args.nms_thresh,
            )
            print(f"  ► Val mAP@0.5 = {val_map:.4f}  (best = {best_map:.4f})")

            if val_map > best_map:
                best_map = val_map
                best_ckpt = {
                    "epoch":   epoch,
                    "model":   model.state_dict(),
                    "best_map": best_map,
                    "config": {
                        "backbone":  args.backbone,
                        "img_size":  args.img_size,
                        "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES,
                    },
                }
                torch.save(best_ckpt, ckpt_dir / "best.pth")
                print(f"  ✓ Saved best model → {ckpt_dir / 'best.pth'}")

    print(f"\nTraining finished.  Best mAP@0.5 = {best_map:.4f}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
