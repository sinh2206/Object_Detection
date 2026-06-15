from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from predict import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VAL_ANNOTATION,
    DEFAULT_VAL_IMAGE_DIR,
    box_iou,
    imread_unicode,
    imwrite_unicode,
    load_ground_truth,
    match_predictions_to_ground_truth,
    score_image_error,
)
from utils.config import CLASS_NAMES

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SUMMARY = DEFAULT_RESULTS_DIR / "hardcase_summary.json"
DEFAULT_PREDICTIONS = Path("val_predictions.json")


def load_annotation_context(annotation_path: Path) -> Tuple[List[str], Dict[str, dict], Dict[str, List[dict]]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    classes = data.get("classes", [])
    class_names = [str(item) for item in classes if isinstance(item, str)]
    if not class_names:
        class_names = list(CLASS_NAMES)

    image_meta: Dict[str, dict] = {}
    for image in data.get("images", []):
        if not isinstance(image, dict):
            continue
        image_id = str(image.get("id", ""))
        if not image_id:
            continue
        image_meta[image_id] = {
            "file_name": str(image.get("file_name", "")),
            "width": int(image.get("width", 0) or 0),
            "height": int(image.get("height", 0) or 0),
        }

    gt_map = load_ground_truth(annotation_path, class_names)
    return class_names, image_meta, gt_map


def load_predictions(predictions_path: Path) -> Dict[str, List[dict]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Predictions file must contain a list: {predictions_path}")

    pred_map: Dict[str, List[dict]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id", ""))
        if not image_id:
            continue
        if image_id in pred_map:
            raise ValueError(f"Duplicate prediction entry for image_id: {image_id}")

        boxes = item.get("boxes", [])
        if not isinstance(boxes, list):
            boxes = []

        cleaned_boxes: List[dict] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            bbox = box.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                confidence = float(box.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            cleaned_boxes.append(
                {
                    "class": str(box.get("class", "")),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        cleaned_boxes.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        pred_map[image_id] = cleaned_boxes

    return pred_map


def clean_hardcase_images(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    for path in results_dir.iterdir():
        if not path.is_file():
            continue
        if path.name == "hardcase_summary.json":
            continue
        if path.name.startswith("hardcase_") and path.suffix.lower() in VALID_EXTS:
            path.unlink()


def draw_header(image, text: str) -> None:
    height, width = image.shape[:2]
    bar_h = 30
    cv2.rectangle(image, (0, 0), (width, bar_h), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _round_half_up(value: float) -> int:
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def _normalize_bbox_for_draw(bbox, image_shape):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return None

    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    ix1 = max(0, min(w - 1, _round_half_up(x1)))
    iy1 = max(0, min(h - 1, _round_half_up(y1)))
    ix2 = max(0, min(w - 1, _round_half_up(x2)))
    iy2 = max(0, min(h - 1, _round_half_up(y2)))

    if ix2 <= ix1:
        if ix1 < w - 1:
            ix2 = ix1 + 1
        else:
            return None
    if iy2 <= iy1:
        if iy1 < h - 1:
            iy2 = iy1 + 1
        else:
            return None

    return [int(ix1), int(iy1), int(ix2), int(iy2)]


def _draw_labeled_box(image_bgr, bbox, label: str, color, thickness: int = 2) -> None:
    norm = _normalize_bbox_for_draw(bbox, image_bgr.shape)
    if norm is None:
        return

    x1, y1, x2, y2 = norm
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    text_x = x1
    text_y = y1 - 4
    if text_y - th - baseline < 0:
        text_y = min(image_bgr.shape[0] - 4, y2 + th + baseline + 4)

    top = max(0, text_y - th - baseline - 2)
    bottom = min(image_bgr.shape[0] - 1, text_y + baseline + 2)
    right = min(image_bgr.shape[1] - 1, text_x + tw + 4)

    cv2.rectangle(image_bgr, (text_x, top), (right, bottom), color, -1)
    cv2.putText(
        image_bgr,
        label,
        (text_x + 2, bottom - baseline - 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _best_gt_for_pred(pred_box: dict, gt_boxes: Sequence[dict], same_class_only: bool = False) -> Tuple[Optional[int], float, str]:
    pred_class = str(pred_box.get("class", ""))
    best_index: Optional[int] = None
    best_iou = 0.0
    best_class = ""

    for gt_index, gt_box in enumerate(gt_boxes):
        gt_class = str(gt_box.get("class", ""))
        if same_class_only and gt_class != pred_class:
            continue
        iou = box_iou(pred_box.get("bbox", [0, 0, 0, 0]), gt_box.get("bbox", [0, 0, 0, 0]))
        if iou > best_iou:
            best_index = gt_index
            best_iou = float(iou)
            best_class = gt_class

    return best_index, best_iou, best_class


def _best_pred_for_gt(gt_box: dict, pred_boxes: Sequence[dict]) -> Tuple[Optional[int], float, str]:
    gt_class = str(gt_box.get("class", ""))
    best_index: Optional[int] = None
    best_iou = 0.0
    best_class = ""

    for pred_index, pred_box in enumerate(pred_boxes):
        pred_class = str(pred_box.get("class", ""))
        iou = box_iou(gt_box.get("bbox", [0, 0, 0, 0]), pred_box.get("bbox", [0, 0, 0, 0]))
        if iou > best_iou:
            best_index = pred_index
            best_iou = float(iou)
            best_class = pred_class
        elif pred_class == gt_class and best_class != gt_class and iou == best_iou and iou > 0:
            best_index = pred_index
            best_class = pred_class

    return best_index, best_iou, best_class


def analyze_image(
    image_id: str,
    image_meta: dict,
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    class_names: Sequence[str],
    iou_thresh: float,
) -> Optional[dict]:
    pred_is_correct, pred_matched_gt, gt_is_matched = match_predictions_to_ground_truth(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        class_names=class_names,
        iou_thresh=iou_thresh,
    )
    metrics = score_image_error(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        class_names=class_names,
        iou_thresh=iou_thresh,
    )

    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    if fp == 0 and fn == 0:
        return None

    gt_details: List[dict] = []
    pred_details: List[dict] = []
    matched_pairs: List[dict] = []

    for gt_index, gt_box in enumerate(gt_boxes):
        matched_pred_index = next((idx for idx, value in enumerate(pred_matched_gt) if value == gt_index), None)
        best_pred_index, best_iou, best_pred_class = _best_pred_for_gt(gt_box, pred_boxes)
        matched_iou = 0.0
        if matched_pred_index is not None:
            matched_iou = float(box_iou(gt_box.get("bbox", [0, 0, 0, 0]), pred_boxes[matched_pred_index].get("bbox", [0, 0, 0, 0])))

        gt_details.append(
            {
                "index": gt_index,
                "class": str(gt_box.get("class", "")),
                "bbox": list(gt_box.get("bbox", [])),
                "matched": bool(gt_is_matched[gt_index]),
                "matched_prediction_index": matched_pred_index,
                "matched_iou": matched_iou,
                "best_prediction_index": best_pred_index,
                "best_prediction_class": best_pred_class,
                "best_iou": best_iou,
            }
        )

    for pred_index, pred_box in enumerate(pred_boxes):
        matched_gt_index = pred_matched_gt[pred_index]
        matched_iou = 0.0
        if matched_gt_index is not None:
            matched_iou = float(box_iou(pred_box.get("bbox", [0, 0, 0, 0]), gt_boxes[matched_gt_index].get("bbox", [0, 0, 0, 0])))
            matched_pairs.append(
                {
                    "prediction_index": pred_index,
                    "ground_truth_index": matched_gt_index,
                    "class": str(pred_box.get("class", "")),
                    "iou": matched_iou,
                }
            )

        best_gt_index, best_iou, best_gt_class = _best_gt_for_pred(pred_box, gt_boxes, same_class_only=False)
        best_same_class_index, best_same_class_iou, _ = _best_gt_for_pred(pred_box, gt_boxes, same_class_only=True)

        pred_details.append(
            {
                "index": pred_index,
                "class": str(pred_box.get("class", "")),
                "confidence": float(pred_box.get("confidence", 0.0)),
                "bbox": list(pred_box.get("bbox", [])),
                "matched": bool(pred_is_correct[pred_index]),
                "status": "tp" if pred_is_correct[pred_index] else "fp",
                "matched_gt_index": matched_gt_index,
                "matched_iou": matched_iou,
                "best_gt_index": best_gt_index,
                "best_gt_class": best_gt_class,
                "best_iou": best_iou,
                "best_same_class_gt_index": best_same_class_index,
                "best_same_class_iou": best_same_class_iou,
            }
        )

    missed_gt = [item for item in gt_details if not item["matched"]]
    false_positives = [item for item in pred_details if not item["matched"]]
    error_types: List[str] = []
    if fn > 0:
        error_types.append("missing_ground_truth")
    if fp > 0:
        error_types.append("false_positive")

    return {
        "image_id": image_id,
        "file_name": str(image_meta.get("file_name", "")),
        "width": int(image_meta.get("width", 0) or 0),
        "height": int(image_meta.get("height", 0) or 0),
        "error_score": float(metrics["error_score"]),
        "error_ratio": float(metrics["error_ratio"]),
        "tp": int(metrics["tp"]),
        "fp": fp,
        "fn": fn,
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "error_types": error_types,
        "matched_pairs": matched_pairs,
        "ground_truth_boxes": gt_details,
        "predicted_boxes": pred_details,
        "missed_ground_truth_boxes": missed_gt,
        "false_positive_boxes": false_positives,
        "rendered_image": "",
        "rendered": False,
    }


def draw_error_image(image_bgr, item: dict) -> Any:
    out = image_bgr.copy()

    for gt_box in item.get("missed_ground_truth_boxes", []):
        cls_name = str(gt_box.get("class", ""))
        _draw_labeled_box(
            out,
            gt_box.get("bbox", [0, 0, 0, 0]),
            f"GT-MISS:{cls_name}",
            (140, 140, 140),
            thickness=2,
        )

    for pred_box in item.get("predicted_boxes", []):
        cls_name = str(pred_box.get("class", ""))
        conf = float(pred_box.get("confidence", 0.0))
        matched = bool(pred_box.get("matched", False))
        if matched:
            label = f"PD:{cls_name}:{conf:.2f}"
            color = (40, 220, 70)
        else:
            label = f"PD-FP:{cls_name}:{conf:.2f}"
            best_iou = float(pred_box.get("best_same_class_iou", 0.0))
            if best_iou <= 0:
                best_iou = float(pred_box.get("best_iou", 0.0))
            if best_iou > 0:
                label = f"{label} iou={best_iou:.2f}"
            color = (0, 0, 255)
        _draw_labeled_box(out, pred_box.get("bbox", [0, 0, 0, 0]), label, color, thickness=2)

    return out


def build_hardcase_report(
    predictions_path: Path,
    image_dir: Path,
    annotation_path: Path,
    results_dir: Path,
    summary_path: Path,
    limit: int = 0,
    iou_thresh: float = 0.5,
) -> Tuple[int, int, Path]:
    class_names, image_meta, gt_map = load_annotation_context(annotation_path)
    pred_map = load_predictions(predictions_path)

    items: List[dict] = []
    for image_id, meta in image_meta.items():
        gt_boxes = gt_map.get(image_id, [])
        pred_boxes = pred_map.get(image_id, [])
        item = analyze_image(
            image_id=image_id,
            image_meta=meta,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            class_names=class_names,
            iou_thresh=iou_thresh,
        )
        if item is not None:
            items.append(item)

    items.sort(
        key=lambda item: (
            -float(item.get("error_ratio", 0.0)),
            -float(item.get("error_score", 0.0)),
            -int(item.get("fn", 0)),
            -int(item.get("fp", 0)),
            str(item.get("image_id", "")),
        )
    )

    for rank, item in enumerate(items, start=1):
        image_id = str(item.get("image_id", ""))
        item["rank"] = rank
        item["rendered_image"] = f"hardcase_{rank:03d}_{image_id}"

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    clean_hardcase_images(results_dir)

    render_items = items if limit <= 0 else items[:limit]
    saved = 0

    for item in render_items:
        image_id = str(item.get("image_id", ""))
        image_path = image_dir / image_id
        image = imread_unicode(image_path)
        if image is None:
            item["render_error"] = f"Could not read image: {image_path}"
            continue

        vis = draw_error_image(image, item)
        draw_header(
            vis,
            (
                f"rank={int(item.get('rank', 0))} image={image_id} "
                f"err={float(item.get('error_score', 0.0)):.3f} "
                f"ratio={float(item.get('error_ratio', 0.0)):.3f} "
                f"tp={int(item.get('tp', 0))} fp={int(item.get('fp', 0))} fn={int(item.get('fn', 0))}"
            ),
        )

        out_path = results_dir / str(item.get("rendered_image", image_id))
        if imwrite_unicode(out_path, vis):
            item["rendered"] = True
            saved += 1
        else:
            item["render_error"] = f"Could not write image: {out_path}"

    summary_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(items), saved, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find every incorrect validation image, render error boxes, and export a detailed hardcase summary."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR)
    parser.add_argument("--val_annotation", type=Path, default=DEFAULT_VAL_ANNOTATION)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render only the first N error images. 0 = render all. The summary always keeps all incorrect images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.val_annotation.exists():
        raise FileNotFoundError(f"Validation annotation not found: {args.val_annotation}")

    item_count, saved_count, summary_path = build_hardcase_report(
        predictions_path=args.predictions,
        image_dir=args.image_dir,
        annotation_path=args.val_annotation,
        results_dir=args.results_dir,
        summary_path=args.summary,
        limit=max(0, int(args.limit)),
        iou_thresh=float(args.iou_thresh),
    )

    print(f"Incorrect images found: {item_count}")
    print(f"Rendered images: {saved_count}")
    print(f"Predictions: {args.predictions}")
    print(f"Summary: {summary_path}")
    print(f"Output dir: {args.results_dir}")


if __name__ == "__main__":
    main()
