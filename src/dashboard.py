# src/dashboard.py
import streamlit as st
import cv2
import numpy as np
from detector import detect, draw_boxes

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("YOLO demo: run detection on a video file or your webcam.")

# --- sidebar controls ---
mode = st.sidebar.selectbox("Source", ["Video file (MP4)", "Webcam"])
conf = st.sidebar.slider("Confidence", 0.10, 0.80, 0.35, 0.05)
max_frames = st.sidebar.number_input("Max frames to process", 50, 2000, 300, 50)

start = st.button("▶️ Start")
frame_area = st.empty()

def process_stream(cap):
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect(frame, conf=conf)
        frame_out = draw_boxes(frame, boxes)
        # convert BGR -> RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        frame_area.image(frame_rgb, channels="RGB")
        count += 1

if start:
    if mode == "Video file (MP4)":
        # change to your file name if different
        path = "data/sample.mp4"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            st.error("Couldn't open data/sample.mp4. Place a test video at data/sample.mp4.")
        else:
            process_stream(cap)
            cap.release()
    else:  # Webcam
        cap = cv2.VideoCapture(0)  # 0 = default camera
        if not cap.isOpened():
            st.error("Webcam not available or permission denied.")
        else:
            process_stream(cap)
            cap.release()

st.success("✅ Ready. Click ▶️ Start to run.")
