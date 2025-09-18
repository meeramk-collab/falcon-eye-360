# src/dashboard.py  — webcam-only YOLO
import streamlit as st, cv2
from detector import detect, draw_boxes

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("YOLO demo: webcam only (no video files)")

# sidebar controls
conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
cam_index = st.sidebar.selectbox("Camera index", [0, 1], index=0)
max_frames = st.sidebar.number_input("Max frames", 50, 3000, 600, 50)

start = st.button("▶️ Start webcam")
frame_area = st.empty()

def run(cap):
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            st.error("Can't read from webcam."); break
        boxes = detect(frame, conf=conf)
        frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2.COLOR_BGR2RGB)
        frame_area.image(frame, channels="RGB")
        count += 1

if start:
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error(
            f"Webcam {cam_index} not available or permission denied. "
            "Enable Camera for Terminal in System Settings."
        )
    else:
        run(cap)
        cap.release()
