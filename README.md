# Object Detection with YOLOv8s

This project trains an object detector with `yolov8s.pt` on the dataset stored in `public/`.

## Dataset layout

- `public/train/images`: training images
- `public/val/images`: validation images
- `public/annotations/train.json`: training annotations
- `public/annotations/val.json`: validation annotations
- `public/classes.json`: class names

The annotation files use project-specific `xyxy` boxes, so the notebook converts them to YOLO labels before training.

## Recommended workflow

Run [Object_Detection.ipynb](Object_Detection.ipynb) on Google Colab with a T4 GPU runtime.

The notebook will:

1. clone the repository into Colab
2. install Ultralytics YOLO
3. convert `train.json` and `val.json` to YOLO label files
4. train `yolov8s.pt`
5. validate the best checkpoint
6. export validation predictions back to the project JSON format
7. copy `yolov8s.pt`, `best.pt`, and `last.pt` into `models/`

## Training artifacts

After the notebook finishes, the main outputs are:

- `models/yolov8s.pt`
- `models/best.pt`
- `models/last.pt`
- `public/dataset_yolov8.yaml`
- `val_predictions.json`
- `val_metrics.json`

The full Ultralytics training logs and plots are stored under `runs/`.
