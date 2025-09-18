# src/dashboard.py
import streamlit as st, cv2
from detector import detect, draw_boxes
# (optional) from tracker import track; from events import update_and_check

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("YOLO demo: webcam mode")

conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
start = st.button("▶️ Start")
frame_area = st.empty()
# alerts = st.container()

def run(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        boxes = detect(frame, conf=conf)
        # tracked = track(boxes)
        # for inc in update_and_check(tracked, fps=fps)[:3]:
        #     alerts.warning(f"ALERT: {inc['type']} → {inc['severity']}")
        frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2.COLOR_BGR2RGB)
        frame_area.image(frame, channels="RGB")

if start:
    cap = None
    for idx in (0, 1):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c; break
    if not cap:
        st.error("Webcam not available or permission denied. Enable camera for Terminal in System Settings.")
    else:
        run(cap); cap.release()
