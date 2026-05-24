from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.config import CLASS_NAMES, IMG_SIZE, MEAN, NUM_CLASSES, PERSON_CLASS_NAME, STD, STRIDES
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    boxes: List[List[float]]
    labels: List[int]


def compute_class_weights(
    samples: List[Sample],
    num_classes: int,
    class_balance: float = 0.35,
    person_boost: float = 1.25,
    class_names: Optional[List[str]] = None,
) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for s in samples:
        for y in s.labels:
            if 0 <= int(y) < num_classes:
                counts[int(y)] += 1.0
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    inv = 1.0 / np.sqrt(freq + 1e-12)  # rare class > 1
    base = np.ones_like(inv)
    alpha = float(np.clip(class_balance, 0.0, 1.0))
    weights = (1.0 - alpha) * base + alpha * inv

    if class_names is None:
        class_names = CLASS_NAMES
    if PERSON_CLASS_NAME in class_names:
        pid = class_names.index(PERSON_CLASS_NAME)
        weights[pid] *= float(max(0.5, person_boost))

    weights = weights / max(weights.mean(), 1e-12)
    weights = np.clip(weights, 0.6, 2.5)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_sample_weights(
    samples: List[Sample],
    class_weights: torch.Tensor,
    person_priority: float = 1.15,
    class_names: Optional[List[str]] = None,
) -> torch.Tensor:
    cw = class_weights.cpu().numpy().astype(np.float64)
    ws = np.ones((len(samples),), dtype=np.float64)
    person_idx = -1
    if class_names is None:
        class_names = CLASS_NAMES
    if PERSON_CLASS_NAME in class_names:
        person_idx = class_names.index(PERSON_CLASS_NAME)
    for i, s in enumerate(samples):
        if len(s.labels) == 0:
            ws[i] = 1.0
            continue
        uniq = np.unique(np.asarray(s.labels, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < len(cw))]
        if uniq.size == 0:
            ws[i] = 1.0
            continue
        # Emphasize images containing rare classes.
        w = float(np.max(cw[uniq]))
        if person_idx >= 0 and person_idx in uniq:
            w *= float(max(0.5, person_priority))
        ws[i] = w
    ws = ws / max(ws.mean(), 1e-12)
    ws = np.clip(ws, 0.3, 4.0)
    return torch.as_tensor(ws, dtype=torch.double)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_samples(annotation_path: Path, image_dir: Path, class_names: Optional[List[str]] = None) -> Tuple[List[Sample], List[str]]:
    data = load_annotation(annotation_path)
    json_classes = data.get("classes", [])
    classes = class_names if class_names is not None else (json_classes if json_classes else CLASS_NAMES)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_by_image: Dict[str, List[dict]] = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        ann_by_image.setdefault(image_id, []).append(ann)

    samples: List[Sample] = []
    for im in images:
        image_id = str(im.get("id"))
        file_name = Path(str(im.get("file_name", image_id))).name
        image_path = image_dir / file_name
        if not image_path.exists():
            fallback = image_dir / image_id
            if fallback.exists():
                image_path = fallback
            else:
                continue

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in ann_by_image.get(image_id, []):
            cls_name = ann.get("class")
            if cls_name not in class_to_idx:
                continue
            x1, y1, x2, y2 = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_to_idx[cls_name])

        samples.append(Sample(image_id=image_id, image_path=image_path, boxes=boxes, labels=labels))

    if not samples:
        raise ValueError(f"No valid samples from {annotation_path} with image_dir={image_dir}")
    return samples, classes


def make_pad_if_needed(img_size: int):
    params = inspect.signature(A.PadIfNeeded.__init__).parameters
    kwargs = {
        "min_height": img_size,
        "min_width": img_size,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 1.0,
    }
    if "fill" in params:
        kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in params:
            kwargs["fill_mask"] = 0
    elif "value" in params:
        kwargs["value"] = (114, 114, 114)
        if "mask_value" in params:
            kwargs["mask_value"] = 0
    return A.PadIfNeeded(**kwargs)


def get_train_transforms(img_size: int) -> A.Compose:
    affine_params = inspect.signature(A.Affine.__init__).parameters
    affine_kwargs = dict(
        scale=(0.9, 1.1),
        translate_percent=(-0.08, 0.08),
        rotate=(-7, 7),
        shear=(-2, 2),
        p=0.3,
    )
    if "border_mode" in affine_params:
        affine_kwargs["border_mode"] = cv2.BORDER_CONSTANT
    elif "mode" in affine_params:
        affine_kwargs["mode"] = cv2.BORDER_CONSTANT

    if "fill" in affine_params:
        affine_kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in affine_params:
            affine_kwargs["fill_mask"] = 0
    elif "value" in affine_params:
        affine_kwargs["value"] = (114, 114, 114)
        if "mask_value" in affine_params:
            affine_kwargs["mask_value"] = 0
    elif "cval" in affine_params:
        affine_kwargs["cval"] = (114, 114, 114)
        if "cval_mask" in affine_params:
            affine_kwargs["cval_mask"] = 0

    affine = A.Affine(**affine_kwargs)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08, p=0.6),
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(70, 145), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.18, p=1.0),
                ],
                p=0.45,
            ),
            affine,
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=4.0,
            min_visibility=0.2,
            clip=True,
        ),
    )


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=1.0,
            min_visibility=0.0,
            clip=True,
        ),
    )


class DetectionDataset(Dataset):
    def __init__(self, samples: List[Sample], transforms: A.Compose):
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = imread_unicode(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [list(b) for b in sample.boxes]
        class_labels = list(sample.labels)

        transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        img_t = transformed["image"].float()
        out_boxes = transformed["bboxes"]
        out_labels = transformed["class_labels"]

        if len(out_boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.as_tensor(np.array(out_boxes, dtype=np.float32))
            labels_t = torch.as_tensor(np.array(out_labels, dtype=np.int64))

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": sample.image_id,
        }
        return img_t, target


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = [x[1] for x in batch]
    return images, targets


def move_targets_to_device(targets: List[dict], device: torch.device) -> List[dict]:
    out = []
    for t in targets:
        out.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"],
            }
        )
    return out


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in running}
    return {k: v / steps for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in running}
    return {k: v / steps for k, v in running.items()}


def build_optimizer(model: AnchorFreeDetector, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone_fpn.parameters())
    head_params = list(model.head_s16.parameters()) + list(model.head_s32.parameters())

    return AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    classes: List[str],
    img_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "classes": classes,
            "img_size": img_size,
            "strides": STRIDES,
        },
        str(path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-free detector (ResNet18 + 2-level FPN).")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--class_balance", type=float, default=0.35)
    parser.add_argument("--person_boost", type=float, default=1.25)
    parser.add_argument("--person_priority", type=float, default=1.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--center_radius", type=float, default=1.5)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(args.img_size))
    val_ds = DetectionDataset(val_samples, transforms=get_val_transforms(args.img_size))

    class_weights = compute_class_weights(
        train_samples,
        num_classes=num_classes,
        class_balance=float(args.class_balance),
        person_boost=float(args.person_boost),
        class_names=classes,
    )
    train_sampler = None
    if not args.no_balanced_sampling:
        sample_weights = build_sample_weights(
            train_samples,
            class_weights,
            person_priority=float(args.person_priority),
            class_names=classes,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AnchorFreeDetector(num_classes=num_classes, pretrained=True).to(device)
    criterion = DetectionLoss(
        num_classes=num_classes,
        strides=STRIDES,
        class_weights=class_weights,
        label_smoothing=float(args.label_smoothing),
        center_radius=float(args.center_radius),
        use_scale_ranges=not args.no_scale_ranges,
    ).to(device)
    optimizer = build_optimizer(model, lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device}, AMP: {amp_enabled}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Class weights: {class_weights.tolist()}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_metrics = validate_one_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, reg={train_metrics['loss_reg']:.4f}, ctr={train_metrics['loss_ctr']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                classes=classes,
                img_size=args.img_size,
            )
            print(f"Saved best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")

    if not best_path.exists():
        save_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
