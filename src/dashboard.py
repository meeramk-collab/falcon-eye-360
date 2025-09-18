import streamlit as st, cv2
from detector import detect, draw_boxes   # your existing detector.py
from tracker import track
from events import update_and_check

st.title("Falcon Eye 360")
source = st.sidebar.selectbox("Source", ["Webcam", "Video file"])
video_path = st.sidebar.text_input("Video path", "data/sample.mp4")
conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
start = st.button("▶️ Start")
frame_area = st.empty()
alerts = st.container()

def run(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        boxes = detect(frame, conf=conf)
        tracked = track(boxes)
        for inc in update_and_check(tracked, fps=fps)[:3]:
            alerts.warning(f"ALERT: {inc['type']} → {inc['severity']}")
        frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2._
