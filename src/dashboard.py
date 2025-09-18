# src/dashboard.py  — webcam-only with tracker + alerts
import streamlit as st, cv2
from detector import detect, draw_boxes
from tracker import track
from events import update_and_check

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("Webcam-only • YOLO + ID tracking + alert rules")

# sidebar controls
conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
cam_index = st.sidebar.selectbox("Camera index", [0, 1], index=0)
max_frames = st.sidebar.number_input("Max frames", 50, 3000, 800, 50)

start = st.button("▶️ Start webcam")
frame_area = st.empty()
alerts_box = st.container()

def run(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            st.error("Can't read from webcam."); break

        # detect -> track -> check incidents
        boxes = detect(frame, conf=conf)
        tracked = track(boxes)
        incidents = update_and_check(tracked, fps=fps)

        # show alerts (limit spam)
        if incidents:
            with alerts_box:
                for inc in incidents[:3]:
                    st.warning(f"ALERT: {inc['type']} → {inc['severity']}")

        # draw boxes and display
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
