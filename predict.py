"""
predict.py  –  Object Detection Inference Script

Usage:
    python predict.py \
        --image_dir /path/to/images \
        --output    predictions.json \
        [--checkpoint ./models/best.pth] \
        [--conf_thresh 0.30] \
        [--nms_thresh  0.50] \
        [--img_size    448]

Output format (predictions.json):
[
  {
    "image_id": "img_xxxx.jpg",
    "boxes": [
      {"class": "person", "confidence": 0.91, "bbox": [48, 72, 210, 356]},
      ...
    ]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLASS_NAMES,
    CONF_THRESH,
    FEATURE_STRIDE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    VALID_IMAGE_EXTS,
)
from utils.inference import predict_single_image
from utils.model import AnchorFreeDetector


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference and output predictions.json")
    p.add_argument("--image_dir",  required=True, help="Directory of images to predict")
    p.add_argument("--output",     default="predictions.json",
                   help="Output JSON file path")
    p.add_argument("--checkpoint", default="./models/best.pth",
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--backbone",   default="resnet34",
                   choices=["resnet34", "resnet18"])
    p.add_argument("--img_size",   type=int,   default=IMG_SIZE)
    p.add_argument("--conf_thresh",type=float, default=CONF_THRESH)
    p.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    p.add_argument("--max_det",    type=int,   default=100,
                   help="Max detections per image")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, backbone: str, device: torch.device) -> AnchorFreeDetector:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train.py first or specify --checkpoint explicitly."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # Checkpoint may contain 'config' dict from train.py
    if "config" in ckpt:
        cfg = ckpt["config"]
        backbone    = cfg.get("backbone", backbone)
        num_classes = cfg.get("num_classes", NUM_CLASSES)
    else:
        num_classes = NUM_CLASSES

    model = AnchorFreeDetector(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,   # weights come from checkpoint
    ).to(device)

    state = ckpt.get("model", ckpt)   # support bare state-dict saves
    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    best_map = ckpt.get("best_map", "?")
    print(f"Loaded checkpoint: {ckpt_path}  (epoch={epoch}, best_mAP={best_map})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_paths(image_dir: str) -> List[Path]:
    folder = Path(image_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )
    return paths


@torch.no_grad()
def run_inference(
    model: AnchorFreeDetector,
    image_paths: List[Path],
    device: torch.device,
    img_size: int,
    conf_thresh: float,
    nms_thresh: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    n = len(image_paths)

    for i, path in enumerate(image_paths):
        image_id = path.name
        try:
            dets = predict_single_image(
                model=model,
                image_path=path,
                device=device,
                img_size=img_size,
                stride=float(FEATURE_STRIDE),
                conf_thresh=conf_thresh,
                nms_iou_thresh=nms_thresh,
                class_names=CLASS_NAMES,
                max_detections=max_det,
            )
            # Round coordinates to 2 decimal places
            for d in dets:
                d["bbox"]       = [round(float(v), 2) for v in d["bbox"]]
                d["confidence"] = round(float(d["confidence"]), 4)
        except Exception as exc:
            print(f"[WARN] Failed {image_id}: {exc}")
            dets = []

        results.append({"image_id": image_id, "boxes": dets})

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Processed {i+1}/{n} images …")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.backbone, device)

    # Collect images
    image_paths = collect_image_paths(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    if len(image_paths) == 0:
        print("No images found – writing empty predictions file.")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return

    # Run inference
    print("Running inference …")
    results = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        max_det=args.max_det,
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_boxes = sum(len(r["boxes"]) for r in results)
    print(f"\nDone.  {len(results)} images  |  {total_boxes} total detections")
    print(f"Predictions saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
