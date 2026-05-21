from __future__ import annotations

import argparse
import colorsys
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils.config import CLASS_NAMES, CONF_THRESH, FEATURE_STRIDE, IMG_SIZE, NMS_IOU_THRESH
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
    p.add_argument("--topk_per_class", type=int, default=400)
    p.add_argument("--max_detections", type=int, default=80)
    p.add_argument("--min_box_size", type=float, default=3.0)
    p.add_argument("--no_tta_flip", action="store_true")

    p.add_argument("--visualize_samples", type=int, default=50)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Prefer EMA weights for inference when available.
        state_dict = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
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


def _letterbox_for_vis(image_path: Path, img_size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    scale = min(float(img_size) / float(orig_w), float(img_size) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), color=(114, 114, 114))
    pad_x = int((img_size - new_w) // 2)
    pad_y = int((img_size - new_h) // 2)
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.uint8).copy()
    meta = {"scale": scale, "pad_x": float(pad_x), "pad_y": float(pad_y)}
    return arr, meta


def _build_color_map(class_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    color_map: Dict[str, Tuple[int, int, int]] = {}
    n = max(1, len(class_names))
    for idx, cls_name in enumerate(class_names):
        h = idx / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
        color_map[cls_name] = (int(255 * b), int(255 * g), int(255 * r))  # BGR for OpenCV
    return color_map


def _draw_box(
    canvas: np.ndarray,
    box: List[float],
    cls_name: str,
    conf: float,
    meta: Dict[str, float],
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(round(x1 * meta["scale"] + meta["pad_x"]))
    y1 = int(round(y1 * meta["scale"] + meta["pad_y"]))
    x2 = int(round(x2 * meta["scale"] + meta["pad_x"]))
    y2 = int(round(y2 * meta["scale"] + meta["pad_y"]))

    h, w = canvas.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"{cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_text_top = max(0, y1 - th - 6)
    x_text_right = min(w - 1, x1 + tw + 6)
    cv2.rectangle(canvas, (x1, y_text_top), (x_text_right, y1), color, -1)
    cv2.putText(
        canvas,
        label,
        (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def export_random_visualizations(
    predictions: List[Dict[str, Any]],
    image_dir: str,
    class_names: List[str],
    img_size: int,
    results_dir: str,
    num_samples: int = 50,
    seed: int = 42,
) -> int:
    if num_samples <= 0 or not predictions:
        return 0

    image_root = Path(image_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_entries = [entry for entry in predictions if (image_root / entry["image_id"]).exists()]
    if not valid_entries:
        return 0

    rng = random.Random(seed)
    k = min(num_samples, len(valid_entries))
    sampled = rng.sample(valid_entries, k=k)
    color_map = _build_color_map(class_names)

    for idx, entry in enumerate(sampled, start=1):
        image_id = entry["image_id"]
        canvas, meta = _letterbox_for_vis(image_root / image_id, img_size=img_size)
        for box in entry.get("boxes", []):
            cls_name = str(box.get("class", "unknown"))
            conf = float(box.get("confidence", 0.0))
            bbox = box.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            color = color_map.get(cls_name, (0, 255, 255))
            _draw_box(canvas, bbox, cls_name, conf, meta, color)

        save_path = out_dir / f"sample_{idx:03d}_{image_id}"
        cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return k


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
        topk_per_class=args.topk_per_class,
        max_detections=args.max_detections,
        min_box_size=args.min_box_size,
        tta_flip=not args.no_tta_flip,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        f.write("\n")

    num_vis = export_random_visualizations(
        predictions=predictions,
        image_dir=args.image_dir,
        class_names=meta["class_names"],
        img_size=img_size,
        results_dir=args.results_dir,
        num_samples=args.visualize_samples,
        seed=args.seed,
    )

    print(f"Saved predictions for {len(predictions)} images to {out_path}")
    print(f"Saved {num_vis} random annotated processed images to {Path(args.results_dir)}")


if __name__ == "__main__":
    main()
