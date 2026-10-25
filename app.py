from __future__ import annotations

import io
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "VisionForge AI"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=f"{APP_TITLE} | YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(46, 91, 255, 0.14), transparent 28%),
            radial-gradient(circle at 85% 5%, rgba(126, 87, 194, 0.12), transparent 30%),
            #07111f;
        color: #eef4ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1628 0%, #0d1c31 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        color: #eef4ff;
    }

    .vf-hero {
        padding: 24px 26px;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, rgba(46,91,255,0.18), rgba(126,87,194,0.10));
        box-shadow: 0 12px 40px rgba(0,0,0,0.22);
        margin-bottom: 18px;
    }

    .vf-title {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
        color: #ffffff;
    }

    .vf-subtitle {
        margin-top: 8px;
        color: #aebdd2;
        font-size: 1rem;
    }

    .vf-badge {
        display: inline-block;
        margin-top: 14px;
        padding: 7px 11px;
        border-radius: 999px;
        color: #dfe8ff;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.82rem;
    }

    .metric-card {
        padding: 16px 18px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 108px;
    }

    .metric-label {
        color: #8fa2bc;
        font-size: 0.83rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 1.72rem;
        font-weight: 800;
    }

    .section-card {
        padding: 18px;
        border-radius: 20px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 14px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 12px !important;
        min-height: 42px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(255,255,255,0.18);
        border-radius: 16px;
        padding: 8px;
        background: rgba(255,255,255,0.02);
    }

    .small-note {
        color: #93a6bd;
        font-size: 0.86rem;
    }

    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return YOLO(model_name)


def get_detection_dataframe(result) -> pd.DataFrame:
    rows = []
    names = result.names
    if result.boxes is None:
        return pd.DataFrame(columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        rows.append(
            {
                "Class": names.get(cls_id, str(cls_id)),
                "Confidence": round(conf, 4),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    total = len(df)
    unique = int(df["Class"].nunique()) if total else 0
    avg_conf = float(df["Confidence"].mean()) if total else 0.0
    top_class = df["Class"].value_counts().index[0] if total else "—"
    return total, unique, avg_conf, top_class


def run_image_inference(image: Image.Image, model, conf: float, iou: float, imgsz: int):
    start = time.perf_counter()
    results = model.predict(
        source=np.array(image.convert("RGB")),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    df = get_detection_dataframe(result)
    return annotated_rgb, df, elapsed


def process_video(input_path: str, model, conf: float, iou: float, imgsz: int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"detected_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    progress = st.progress(0.0, text="Processing video...")
    status = st.empty()

    aggregate_rows = []
    start = time.perf_counter()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        writer.write(annotated)

        df = get_detection_dataframe(result)
        if not df.empty:
            df.insert(0, "Frame", frame_idx)
            aggregate_rows.append(df)

        frame_idx += 1
        if frame_count > 0:
            p = min(frame_idx / frame_count, 1.0)
            progress.progress(p, text=f"Processing video... {frame_idx}/{frame_count} frames")
        if frame_idx % 15 == 0:
            status.caption(f"Processed {frame_idx} frames")

    cap.release()
    writer.release()
    progress.empty()
    status.empty()

    elapsed = time.perf_counter() - start
    all_df = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame(
        columns=["Frame", "Class", "Confidence", "X1", "Y1", "X2", "Y2"]
    )
    return out_path, all_df, elapsed, frame_idx


def show_results(annotated_rgb, df: pd.DataFrame, elapsed: float):
    total, unique, avg_conf, top_class = summarize(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Objects Detected", str(total))
    with c2: card("Unique Classes", str(unique))
    with c3: card("Average Confidence", f"{avg_conf*100:.1f}%")
    with c4: card("Most Frequent", str(top_class))

    left, right = st.columns([1.55, 1.0], gap="large")
    with left:
        st.markdown("### Annotated Result")
        st.image(annotated_rgb, use_container_width=True)

    with right:
        st.markdown("### Detection Report")
        if df.empty:
            st.info("No objects were detected at the selected confidence threshold.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download detection CSV",
                data=csv,
                file_name="detections.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption(f"Inference completed in {elapsed:.3f} seconds.")


st.markdown(
    """
    <div class="vf-hero">
        <div class="vf-title">VisionForge AI</div>
        <div class="vf-subtitle">
            A complete YOLOv8 object-detection studio for images, videos and webcam snapshots.
        </div>
        <div class="vf-badge">Ultralytics YOLOv8 • Real-time CV • CSV Reports • Modern UI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    model_choice = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Larger models are typically more accurate but require more compute.",
    )

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    imgsz = st.select_slider("Inference image size", options=[320, 480, 640, 768, 960], value=640)

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "VisionForge AI detects COCO objects using YOLOv8. "
        "The selected model is downloaded automatically by Ultralytics on first use."
    )

try:
    with st.spinner(f"Loading {model_choice}..."):
        model = load_model(model_choice)
except Exception as exc:
    st.error(
        "Model loading failed. On the first run, YOLO weights may need internet access. "
        f"Details: {exc}"
    )
    st.stop()

tab_img, tab_video, tab_cam, tab_info = st.tabs(
    ["🖼️ Image", "🎬 Video", "📷 Webcam", "📘 Project Info"]
)

with tab_img:
    st.markdown("### Image Object Detection")
    uploaded_image = st.file_uploader(
        "Upload JPG, JPEG, PNG or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        preview_col, action_col = st.columns([1.4, 0.6])
        with preview_col:
            st.image(image, caption="Input image", use_container_width=True)
        with action_col:
            st.write("")
            st.write("")
            detect_btn = st.button("🎯 Run Detection", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if detect_btn:
            with st.spinner("Running YOLOv8 inference..."):
                annotated, df, elapsed = run_image_inference(image, model, conf, iou, imgsz)
            show_results(annotated, df, elapsed)

with tab_video:
    st.markdown("### Video Object Detection")
    st.caption("Upload a video. The app will process every frame and create a detected MP4 plus CSV report.")
    uploaded_video = st.file_uploader(
        "Upload MP4, MOV, AVI or MKV",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("🎬 Process Video", type="primary", use_container_width=True):
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.getbuffer())
                temp_input = tmp.name

            try:
                output_path, video_df, elapsed, frames = process_video(
                    temp_input, model, conf, iou, imgsz
                )
                st.success(f"Finished processing {frames} frames in {elapsed:.1f} seconds.")

                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)

                c1, c2, c3 = st.columns(3)
                with c1: card("Processed Frames", str(frames))
                with c2: card("Total Detections", str(len(video_df)))
                with c3:
                    mean_conf = video_df["Confidence"].mean() * 100 if not video_df.empty else 0
                    card("Average Confidence", f"{mean_conf:.1f}%")

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name=output_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )

                if not video_df.empty:
                    st.dataframe(video_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Video Detection CSV",
                        data=video_df.to_csv(index=False).encode("utf-8"),
                        file_name="video_detections.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            finally:
                try:
                    os.remove(temp_input)
                except OSError:
                    pass

with tab_cam:
    st.markdown("### Webcam Snapshot Detection")
    st.caption(
        "Take a photo using your webcam and run YOLOv8 detection directly inside the app."
    )
    camera_image = st.camera_input("Capture an image")

    if camera_image is not None:
        image = Image.open(camera_image)
        if st.button("📷 Detect Webcam Snapshot", type="primary", use_container_width=True):
            with st.spinner("Analyzing webcam image..."):
                annotated, df, elapsed = run_image_inference(image, model, conf, iou, imgsz)
            show_results(annotated, df, elapsed)

with tab_info:
    st.markdown(
        """
        ## Project Overview

        **VisionForge AI** is a complete object-detection application based on **Ultralytics YOLOv8**.
        It is designed as a portfolio-ready computer vision project with a clean user interface,
        configurable inference parameters and downloadable reports.

        ### Main Features
        - YOLOv8 Nano / Small / Medium / Large / XLarge model selection
        - Image detection
        - Full video processing
        - Webcam snapshot detection
        - Confidence and IoU controls
        - Bounding-box visualization
        - Detection table with coordinates and confidence scores
        - CSV export
        - Processed video export
        - Responsive dark UI
        - Clean project structure suitable for GitHub

        ### COCO Object Classes
        Standard pretrained YOLOv8 models detect 80 everyday classes including people, cars,
        buses, bicycles, dogs, cats, bottles, chairs, laptops, phones and many more.

        ### Custom Model
        If you have your own trained YOLOv8 model, place `best.pt` inside the `models/` folder
        and change the model path in the sidebar/model configuration.
        """
    )
