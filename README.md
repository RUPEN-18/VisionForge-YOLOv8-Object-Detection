# VisionForge AI — YOLOv8 Object Detection Studio

A complete portfolio-ready object detection project built with **Ultralytics YOLOv8**, **Streamlit**, **OpenCV**, and **Python**.

## Features

- Modern dark UI
- YOLOv8n / s / m / l / x model selector
- Image object detection
- Video object detection
- Webcam snapshot detection
- Confidence threshold control
- IoU threshold control
- Image-size control
- Bounding boxes and labels
- Detection count and confidence statistics
- Detection report table
- CSV export
- Processed MP4 export
- Supports custom `best.pt` models

## Project Structure

```text
YOLOv8_Object_Detection_Studio/
├── app.py
├── requirements.txt
├── run_app.bat
├── setup_windows.bat
├── README.md
├── .gitignore
├── models/
│   └── README.txt
├── outputs/
│   └── .gitkeep
└── assets/
    └── README.txt
```

## Windows Setup

### Option 1 — Automatic Setup

Double-click:

```text
setup_windows.bat
```

Then launch:

```text
run_app.bat
```

### Option 2 — Manual Setup

Open Command Prompt inside the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser, normally at:

```text
http://localhost:8501
```

## Important: YOLOv8 Weights

The default model is `yolov8n.pt`.

Ultralytics will automatically download pretrained model weights on the first run.
Therefore, **internet access is normally required once** for a model that is not already cached.

After the model weights are present on your computer, inference can run locally/offline.

## Use Your Own Custom YOLOv8 Model

1. Copy your trained model:
   `best.pt`
2. Put it in:
   `models/best.pt`
3. In `app.py`, you can replace a pretrained name such as:

```python
YOLO("yolov8n.pt")
```

with:

```python
YOLO("models/best.pt")
```

You can also extend the model selector to include your custom path.

## Recommended Python

Python 3.10 or 3.11 is recommended.

## GPU Support

The project works on CPU.

For NVIDIA GPU acceleration, install a CUDA-compatible PyTorch build appropriate for your system, then verify:

```python
import torch
print(torch.cuda.is_available())
```

## GitHub Upload

```bash
git init
git add .
git commit -m "Initial release of VisionForge AI YOLOv8 Object Detection"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Notes

- Larger YOLOv8 models generally need more memory and processing power.
- Video processing speed depends heavily on model size, CPU/GPU and video resolution.
- Generated outputs are stored in `outputs/`.
- Model weights are ignored by Git by default because they can be large.

## Portfolio Description

**VisionForge AI – YOLOv8 Object Detection System**

Developed an end-to-end computer vision application using YOLOv8 for real-time object detection across images, videos, and webcam input. Built a modern Streamlit interface with configurable confidence/IoU thresholds, detection analytics, bounding-box visualization, CSV reporting, and processed-video export. The application supports multiple YOLOv8 model sizes and custom trained weights.
