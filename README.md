# Object Detection trên Google Colab T4

Dự án huấn luyện mô hình phát hiện đối tượng cho 5 lớp: `person`, `car`, `dog`, `cat`, `chair`. Workflow chính được đóng gói trong notebook [obj-detec.ipynb](obj-detec.ipynb) để chạy trực tiếp trên Google Colab GPU T4.

## 1. Chuẩn Bị Colab

Mở [obj-detec.ipynb](obj-detec.ipynb) trên Google Colab, sau đó chọn:

```text
Runtime > Change runtime type > Hardware accelerator > T4 GPU
```

Chạy notebook từ trên xuống. Notebook hiện gồm các bước:

| Cell | Chức năng |
|---:|---|
| 1 | Clone hoặc pull repository vào `/content/O_D` |
| 2 | Kiểm tra GPU, CUDA và phiên bản PyTorch |
| 3 | Cài dependencies từ `requirements.txt` |
| 4 | Train model bằng `train.py` |
| 5 | Predict validation set bằng `predict.py` |
| 6 | Evaluate bằng `public/tools/evaluate_predictions.py` |
| 7 | In nội dung `val_score.json` |
| 8 | Zip project ra `/content/train.zip`, bỏ qua thư mục `public/` |

Notebook đã bỏ bước tạo `results/hardcase_summary.json` và ảnh lỗi trong `results/`.

## 2. Cài Đặt

Notebook sẽ tự chạy:

```python
%pip install -r requirements.txt
```

Các thư viện chính:

```text
torch
torchvision
albumentations
opencv-python
Pillow
numpy
```

Nếu Colab báo lỗi môi trường CUDA, restart runtime rồi chạy lại từ đầu. Colab T4 thường đã có PyTorch CUDA phù hợp nên không cần cài thủ công PyTorch riêng.

## 3. Train Trên GPU T4

Cell train trong notebook:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models \
  --img_size 640 \
  --batch_size 8 \
  --epochs 20 \
  --num_workers 2
```

Nếu T4 bị CUDA out-of-memory:

```bash
# Ưu tiên giảm batch size
--batch_size 4

# Nếu vẫn lỗi, giảm image size
--img_size 576
```

Checkpoint được lưu tại:

```text
models/best.pth
models/last.pth
```

## 4. Predict

Cell predict:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --val_annotation ./public/annotations/val.json \
  --output val_predictions.json \
  --model_path ./models/best.pth \
  --device cuda \
  --batch_size 16
```

Output chính:

```text
val_predictions.json
```

`predict.py` hiện chỉ xuất JSON dự đoán. Script không còn tạo hardcase summary hoặc ảnh lỗi trong `results/`.

## 5. Evaluate

Cell evaluate:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

Output:

```text
val_score.json
```

## 6. Kết Quả Hiện Tại

Checkpoint tốt nhất hiện có:

| Thuộc tính | Giá trị |
|---|---:|
| Epoch | 18 |
| Best validation loss | 0.689164 |
| Image size | 640 |
| Classes | person, car, dog, cat, chair |

Kết quả trên validation set:

| Metric | Giá trị |
|---|---:|
| mAP@0.5 | 0.722545 |
| Performance points | 20 |
| Ground truth boxes | 2021 |
| Predictions | 4053 |
| Micro precision | 0.399457 |
| Micro recall | 0.801089 |

Kết quả theo lớp:

| Class | AP | Recall | Precision | GT | Pred | TP | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 0.782319 | 0.818436 | 0.571150 | 1074 | 1539 | 879 | 660 |
| car | 0.684412 | 0.777385 | 0.254042 | 283 | 866 | 220 | 646 |
| dog | 0.775038 | 0.854369 | 0.385965 | 206 | 456 | 176 | 280 |
| cat | 0.857270 | 0.880682 | 0.593870 | 176 | 261 | 155 | 106 |
| chair | 0.513688 | 0.670213 | 0.203008 | 282 | 931 | 189 | 742 |

## 7. Thống Kê Dataset

| Split | Images | Boxes | person | car | dog | cat | chair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 7500 | 10642 | 5829 | 1339 | 1028 | 833 | 1613 |
| Val | 1500 | 2021 | 1074 | 283 | 206 | 176 | 282 |

## 8. Cấu Trúc Dự Án

```text
O_D/
  obj-detec.ipynb
  train.py
  predict.py
  requirements.txt
  README.md
  main.tex
  public/
    annotations/
      train.json
      val.json
    train/images/
    val/images/
  models/
    best.pth
    last.pth
  utils/
    config.py
    model.py
    loss.py
    nms.py
    runtime.py
    image_ops.py
    process.py
    forecast.py
```

## 9. Các Chỉnh Sửa Chính

`train.py`:

- Train anchor-free detector với ResNet34 + FPN 3 mức.
- Dùng class weights và weighted sampler để xử lý mất cân bằng lớp.
- Dùng AMP trên CUDA, AdamW và CosineAnnealingLR.
- Lưu `models/best.pth` và `models/last.pth`.
- Augmentation đã chỉnh theo hướng car-friendly: bỏ `VerticalFlip`, bỏ `RandomRotate90`, thêm JPEG compression và coarse dropout nhẹ.

`predict.py`:

- Đọc ảnh, letterbox, tăng sáng ảnh tối, batch inference.
- Decode multi-level output, kết hợp class score với centerness, class-wise NMS.
- Áp dụng threshold theo lớp và giới hạn số object mỗi ảnh.
- Suppress một số `chair` nằm trong `person`.
- Đã bỏ toàn bộ phần tạo `hardcase_summary.json` và ảnh hardcase.

`utils/`:

- `config.py`: cấu hình lớp, image size, stride, threshold, class weights, low-light và loss weights.
- `model.py`: ResNet34 backbone, FPN 3 mức, anchor-free heads.
- `loss.py`: focal loss, GIoU loss, centerness loss, target assignment.
- `nms.py`: decode bbox, confidence filtering, NMS, remap bbox về ảnh gốc.
- `runtime.py`: helper chọn device, optimizer, scheduler, checkpoint.
- `image_ops.py`: tăng sáng ảnh low-light bằng CLAHE và gamma.
- `process.py`: tiện ích xử lý annotation, đọc ảnh, resize, mosaic.
- `forecast.py`: head/decoder thử nghiệm cho anchor-free detection.

`obj-detec.ipynb`:

- Đã chuyển sang workflow Google Colab T4.
- Bỏ cell gọi `img_error.py`.
- Bỏ đường dẫn `/kaggle/working`, thay bằng `/content/O_D`.
- Thêm cell kiểm tra GPU và cell in `val_score.json`.

## 10. Đóng Gói Kết Quả

Cell cuối notebook tạo:

```text
/content/train.zip
```

File zip loại bỏ `public/` để giảm dung lượng. Các file quan trọng nên có trong zip:

```text
train.py
predict.py
utils/
models/best.pth
models/last.pth
val_predictions.json
val_score.json
README.md
main.tex
obj-detec.ipynb
```
