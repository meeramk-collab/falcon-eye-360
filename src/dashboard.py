# src/dashboard.py  — Webcam-only YOLO demo for FE360
import time
import streamlit as st
import cv2

from detector import detect, draw_boxes  # <-- your existing robust detector.py

st.title("Falcon Eye 360 – Webcam YOLO")
st.caption("Webcam only. No video files.")

# simple controls
conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
fps_limit = st.sidebar.slider("Max FPS (to save CPU)", 5, 30, 12, 1)

# start/stop state
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
start_click = col1.button("▶ Start webcam")
stop_click  = col2.button("■ Stop")

if start_click:
    st.session_state.running = True
if stop_click:
    st.session_state.running = False

frame_area = st.empty()
status = st.empty()

def open_any_camera():
    """Try common indices (0,1,2) and return an opened capture + index."""
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
    return None, None

if st.session_state.running:
    cap, idx = open_any_camera()
    if not cap:
        st.error("Webcam not available or permission denied.\n"
                 "macOS: Settings → Privacy & Security → Camera → allow Terminal/VS Code.")
        st.stop()

    status.info(f"✅ Using camera index: {idx}")
    last = 0.0
    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                status.error("Can't read from webcam."); break

            # throttle FPS to be nice to your laptop
            now = time.time()
            if now - last < 1.0 / max(1, fps_limit):
                time.sleep(0.001); continue
            last = now

            boxes = detect(frame, conf=conf)
            frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2.COLOR_BGR2RGB)
            frame_area.image(frame, channels="RGB")
    finally:
        cap.release()
        status.info("Webcam released. You can press ▶ Start webcam again.")
import streamlit as st, cv2
from detector import detect, draw_boxes

st.title("Webcam YOLO Test")

start = st.button("▶️ Start webcam")
frame_area = st.empty()

if start:
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        st.error("Webcam not available or permission denied.")
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Can't read from webcam.")
                break
            boxes = detect(frame, conf=0.35)
            frame = draw_boxes(frame, boxes)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_area.image(frame, channels="RGB")
