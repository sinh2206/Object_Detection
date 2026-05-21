from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from utils.config import (
    CLASS_NAMES,
    CONF_THRESH,
    FEATURE_STRIDE,
    IMG_SIZE,
    NMS_IOU_THRESH,
)
from utils.inference import predict_images
from utils.model import AnchorFreeDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with anchor-free detector")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--model_path", default="models/best.pth")
    p.add_argument("--conf_thres", type=float, default=CONF_THRESH)
    p.add_argument("--nms_thres", type=float, default=NMS_IOU_THRESH)
    p.add_argument("--img_size", type=int, default=None)
    return p.parse_args()


def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = ckpt
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError("Unsupported checkpoint format")

    class_names = meta.get("class_names", CLASS_NAMES)
    backbone = meta.get("backbone", "resnet18")

    model = AnchorFreeDetector(
        num_classes=len(class_names),
        backbone_name=backbone,
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    model_meta = {
        "img_size": int(meta.get("img_size", IMG_SIZE)),
        "stride": float(meta.get("stride", FEATURE_STRIDE)),
        "class_names": class_names,
    }
    return model, model_meta


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_model(args.model_path, device)
    img_size = int(args.img_size or meta["img_size"])

    predictions = predict_images(
        model=model,
        image_dir=args.image_dir,
        device=device,
        img_size=img_size,
        stride=float(meta["stride"]),
        conf_thresh=args.conf_thres,
        nms_iou_thresh=args.nms_thres,
        class_names=meta["class_names"],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved predictions for {len(predictions)} images to {out_path}")


if __name__ == "__main__":
    main()
