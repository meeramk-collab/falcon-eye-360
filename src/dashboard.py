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
