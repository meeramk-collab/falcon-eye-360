# src/dashboard.py
import streamlit as st, cv2
from detector import detect, draw_boxes
# (optional) add tracker/events later

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("YOLO demo: webcam mode only")

conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
start = st.button("▶️ Start")
frame_area = st.empty()

def run(cap):
    while True:
        ok, frame = cap.read()
        if not ok: break
        boxes = detect(frame, conf=conf)
        frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2.COLOR_BGR2RGB)
        frame_area.image(frame, channels="RGB")

if start:
    cap = None
    # try both common camera indexes on Mac (built-in can be 0 or 1)
    for idx in (0, 1):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            break
    if not cap:
        st.error("Webcam not available or permission denied. Enable Camera for Terminal in System Settings.")
    else:
        run(cap)
        cap.release()
