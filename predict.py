from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils.config import CLASS_NAMES, CONF_THRESH, IMG_SIZE, MEAN, NMS_IOU_THRESH, NUM_CLASSES, STD
from utils.model import AnchorFreeDetector
from utils.nms import LetterboxMeta, postprocess_batch

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in VALID_EXTS else ".jpg"
    out_path = path if path.suffix.lower() in VALID_EXTS else path.with_suffix(ext)
    ok, enc = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    enc.tofile(str(out_path))
    return True


def letterbox_preprocess(image_bgr: np.ndarray, img_size: int) -> Tuple[torch.Tensor, LetterboxMeta]:
    h, w = image_bgr.shape[:2]
    scale = min(float(img_size) / max(w, 1), float(img_size) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    meta = LetterboxMeta(scale=scale, dx=float(dx), dy=float(dy), orig_w=int(w), orig_h=int(h))
    return tensor, meta


def collect_images(image_dir: Path) -> List[Path]:
    imgs = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return imgs


def load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_topk_images_by_objects(annotation_path: Path, image_dir: Path, top_k: int) -> List[Path]:
    data = load_annotation(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    count_by_image: Dict[str, int] = defaultdict(int)
    for ann in annotations:
        image_id = str(ann.get("image_id", ""))
        if image_id:
            count_by_image[image_id] += 1

    rows: List[Tuple[int, str, Path]] = []
    for im in images:
        image_id = str(im.get("id", ""))
        file_name = Path(str(im.get("file_name", image_id))).name
        count = int(count_by_image.get(image_id, 0))
        p = image_dir / file_name
        if not p.exists():
            p = image_dir / image_id
        if p.exists() and p.suffix.lower() in VALID_EXTS:
            rows.append((count, image_id, p))

    rows.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in rows[: max(0, top_k)]]


def draw_prediction(image_bgr: np.ndarray, boxes: Sequence[dict], class_names: Sequence[str]) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for obj in boxes:
        cls_name = str(obj["class"])
        score = float(obj["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in obj["bbox"]]
        idx = cls_to_idx.get(cls_name, 0)
        color = (
            int((53 * (idx + 1)) % 255),
            int((97 * (idx + 1)) % 255),
            int((193 * (idx + 1)) % 255),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}:{score:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def run_inference(
    model: AnchorFreeDetector,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
    img_size: int,
    conf_thresh: float,
    nms_thresh: float,
    class_names: Sequence[str],
    agnostic_nms_thresh: float,
    cross_class_iou_thresh: float,
    cross_class_contain_thresh: float,
) -> List[dict]:
    results: List[dict] = []
    amp_enabled = device.type == "cuda"

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        metas: List[LetterboxMeta] = []
        image_ids: List[str] = []

        for p in batch_paths:
            image = imread_unicode(p)
            if image is None:
                continue
            tensor, meta = letterbox_preprocess(image, img_size=img_size)
            tensors.append(tensor)
            metas.append(meta)
            image_ids.append(p.name)

        if not tensors:
            continue

        images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)

        batch_results = postprocess_batch(
            outputs=outputs,
            image_ids=image_ids,
            metas=metas,
            class_names=class_names,
            num_classes=len(class_names),
            img_size=img_size,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            reg_decode="auto",
            center_combine="mul",
            min_box_size=2.0,
            agnostic_nms_thresh=agnostic_nms_thresh,
            cross_class_iou_thresh=cross_class_iou_thresh,
            cross_class_contain_thresh=cross_class_contain_thresh,
        )
        results.extend(batch_results)

    return results


def save_preview_images(
    predictions: List[dict],
    image_path_map: Dict[str, Path],
    preview_dir: Path,
    limit: int,
    class_names: Sequence[str],
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for pred in predictions[:limit]:
        image_id = pred["image_id"]
        img_path = image_path_map.get(image_id)
        if img_path is None:
            continue
        image = imread_unicode(img_path)
        if image is None:
            continue

        vis = draw_prediction(image, pred.get("boxes", []), class_names=class_names)
        out_path = preview_dir / image_id
        if imwrite_unicode(out_path, vis):
            saved += 1
    return saved


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[AnchorFreeDetector, List[str], int]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    classes = ckpt.get("classes", CLASS_NAMES)
    img_size = int(ckpt.get("img_size", IMG_SIZE))
    num_classes = len(classes)

    model = AnchorFreeDetector(num_classes=num_classes, pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, list(classes), img_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with anchor-free detector and export JSON.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Predict all images in this folder.")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=Path,
        default=Path("models/best.pth"),
        help="Path to trained model checkpoint (.pth). '--model_path' is kept as a backward-compatible alias.",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument("--agnostic_nms_thresh", type=float, default=0.75)
    parser.add_argument("--cross_class_iou_thresh", type=float, default=0.85)
    parser.add_argument("--cross_class_contain_thresh", type=float, default=0.9)
    parser.add_argument("--preview_dir", type=Path, default=Path("results"))
    parser.add_argument("--preview_count", type=int, default=50)
    parser.add_argument(
        "--top_k_objects",
        type=int,
        default=0,
        help="If >0, select top-K images with most objects from train/val annotations (split ~50/50).",
    )
    parser.add_argument("--train_data", type=Path, default=Path("public/annotations/train.json"))
    parser.add_argument("--val_data", type=Path, default=Path("public/annotations/val.json"))
    parser.add_argument("--train_image_dir", type=Path, default=Path("public/train/images"))
    parser.add_argument("--val_image_dir", type=Path, default=Path("public/val/images"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, ckpt_classes, ckpt_img_size = load_checkpoint_model(args.checkpoint, device=device)
    class_names = ckpt_classes if ckpt_classes else CLASS_NAMES
    img_size = args.img_size if args.img_size > 0 else ckpt_img_size

    image_paths: List[Path] = []
    if int(args.top_k_objects) > 0:
        if not args.train_data.exists() or not args.val_data.exists():
            raise FileNotFoundError("When --top_k_objects > 0, --train_data and --val_data must exist.")
        if not args.train_image_dir.exists() or not args.val_image_dir.exists():
            raise FileNotFoundError("When --top_k_objects > 0, --train_image_dir and --val_image_dir must exist.")

        top_k = int(args.top_k_objects)
        top_k_train = top_k // 2
        top_k_val = top_k - top_k_train
        train_paths = select_topk_images_by_objects(args.train_data, args.train_image_dir, top_k=top_k_train)
        val_paths = select_topk_images_by_objects(args.val_data, args.val_image_dir, top_k=top_k_val)
        image_paths = train_paths + val_paths
    else:
        if args.image_dir is None:
            raise ValueError("Please provide --image_dir, or set --top_k_objects > 0.")
        if not args.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        image_paths = collect_images(args.image_dir)

    if not image_paths:
        raise ValueError("No images selected for inference.")

    # Keep unique image names while preserving first occurrence.
    unique_by_name: Dict[str, Path] = {}
    for p in image_paths:
        unique_by_name.setdefault(p.name, p)
    image_paths = list(unique_by_name.values())

    predictions = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=max(1, args.batch_size),
        img_size=img_size,
        conf_thresh=float(args.conf_thresh),
        nms_thresh=float(args.nms_thresh),
        class_names=class_names,
        agnostic_nms_thresh=float(args.agnostic_nms_thresh),
        cross_class_iou_thresh=float(args.cross_class_iou_thresh),
        cross_class_contain_thresh=float(args.cross_class_contain_thresh),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    image_path_map = {p.name: p for p in image_paths}
    saved = save_preview_images(
        predictions=predictions,
        image_path_map=image_path_map,
        preview_dir=args.preview_dir,
        limit=max(0, args.preview_count),
        class_names=class_names,
    )

    print(f"Device: {device}")
    print(f"Predicted images: {len(predictions)}")
    print(f"Saved JSON: {args.output}")
    print(f"Saved preview images: {saved} -> {args.preview_dir}")


if __name__ == "__main__":
    main()
