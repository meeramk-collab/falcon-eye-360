import streamlit as st, cv2
from detector import detect, draw_boxes

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("YOLO demo: webcam mode")

conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
cam_index = st.sidebar.selectbox("Camera index", [0, 1], index=0)  # pick 0 or 1 here
start = st.button("▶️ Start")
frame_area = st.empty()

def run(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Can't read from webcam."); break
        boxes = detect(frame, conf=conf)
        frame = cv2.cvtColor(draw_boxes(frame, boxes), cv2.COLOR_BGR2RGB)
        frame_area.image(frame, channels="RGB")

if start:
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error(f"Webcam {cam_index} not available or permission denied.")
    else:
        run(cap); cap.release()
