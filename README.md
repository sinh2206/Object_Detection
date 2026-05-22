# Object Detection – Anchor-Free (FCOS-style)

Mô hình phát hiện đối tượng theo kiến trúc **Anchor-Free** (giống FCOS), gồm 5 lớp:
`person`, `car`, `dog`, `cat`, `chair`.

---

## Kiến trúc tổng quan

```
Input (448×448)
    │
    ▼
ResNet-34 Backbone  →  Feature Map (512 × 14 × 14)
    │
    ▼
1×1 Conv  →  256 × 14 × 14
    │
    ├── Classification Head  →  5 × 14 × 14   (per-cell class logits)
    ├── Regression Head      →  4 × 14 × 14   (l, t, r, b distances)
    └── Centerness Head      →  1 × 14 × 14   (center-ness score)
```

**Target encoding** (stride = 32):
- Mỗi ô lưới `(row, col)` có tâm `cx = col*32+16`, `cy = row*32+16`.
- Ô được gán là *Positive* nếu `(cx, cy)` rơi vào bên trong một bounding box.
- Regression target: `l = cx − x_min`, `t = cy − y_min`, `r = x_max − cx`, `b = y_max − cy`.
- Centerness: `sqrt( min(l,r)/max(l,r) × min(t,b)/max(t,b) )`.

**Loss function:**
```
L_total = L_cls (Focal BCE, tất cả ô)
        + L_reg (GIoU loss, chỉ ô Positive)
        + L_center (BCE, chỉ ô Positive)
```

**Inference:**
```
score = sigmoid(cls) × sigmoid(centerness)
→ lọc score > conf_thresh
→ khôi phục bbox: x1 = cx-l, y1 = cy-t, x2 = cx+r, y2 = cy+b
→ rescale về kích thước ảnh gốc
→ Per-class NMS
```

---

## Cài đặt môi trường

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv && source venv/bin/activate   # Linux/Mac
# hoặc
python -m venv venv && venv\Scripts\activate      # Windows

# Cài thư viện
pip install -r requirements.txt
```

---

## Cấu trúc thư mục

```
<my_submission>/
├── public/               ← dữ liệu (không nộp lại)
│   ├── train/images/
│   ├── val/images/
│   └── annotations/
│       ├── train.json
│       └── val.json
├── models/               ← checkpoints (best.pth được lưu ở đây)
├── utils/
│   ├── __init__.py
│   ├── anchor_utils.py   ← grid centers, GIoU, centerness
│   ├── augmentations.py  ← Albumentations pipelines
│   ├── config.py         ← tất cả hằng số
│   ├── dataset.py        ← DetectionDataset + target generator
│   ├── inference.py      ← decode + NMS + rescale
│   ├── loss.py           ← Focal BCE + GIoU + centerness BCE
│   ├── metrics.py        ← evaluate_map (mAP@0.5)
│   └── model.py          ← AnchorFreeDetector (ResNet34 backbone)
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

---

## Huấn luyện

```bash
python train.py \
  --train_data    ./public/annotations/train.json \
  --val_data      ./public/annotations/val.json \
  --image_dir     ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### Các tham số tuỳ chọn

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 80 | Số epoch huấn luyện |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-3 | Learning rate ban đầu (head) |
| `--backbone` | resnet34 | `resnet34` hoặc `resnet18` |
| `--img_size` | 448 | Kích thước ảnh đầu vào |
| `--workers` | 4 | Số worker DataLoader |
| `--resume` | None | Tiếp tục từ checkpoint |
| `--no_pretrained` | False | Không dùng pretrained backbone |

Mô hình tốt nhất (theo val mAP@0.5) được lưu tại `./models/best.pth`.

---

## Suy luận (Inference)

```bash
python predict.py \
  --image_dir /path/to/images \
  --output    predictions.json
```

### Các tham số tuỳ chọn

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--checkpoint` | `./models/best.pth` | Đường dẫn checkpoint |
| `--conf_thresh` | 0.30 | Ngưỡng độ tin cậy |
| `--nms_thresh` | 0.50 | Ngưỡng IoU cho NMS |
| `--img_size` | 448 | Kích thước ảnh đầu vào |

---

## Đánh giá

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions  val_predictions.json \
  --output       val_score.json
```

---

## Vị trí trọng số mô hình

Sau khi huấn luyện, trọng số tốt nhất được lưu tại:
```
./models/best.pth
```
File này chứa: `model` (state_dict), `epoch`, `best_map`, `config`.
