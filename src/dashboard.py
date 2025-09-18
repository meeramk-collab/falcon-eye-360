cat > src/dashboard.py <<'PY'
# webcam-only dashboard (no video files)
import streamlit as st, cv2
from detector import detect, draw_boxes

st.title("Falcon Eye 360 - Demo Dashboard")
st.caption("Webcam only — no video files")

conf = st.sidebar.slider("YOLO confidence", 0.10, 0.80, 0.35, 0.05)
cam_index = st.sidebar.selectbox("Camera index", [0, 1], index=0)
max_frames = st.sidebar.number_input("Max frames", 50, 3000, 800, 50)

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
        st.error("Webcam not available or permission denied. "
                 "System Settings → Privacy & Security → Camera → enable Terminal.")
    else:
        run(cap); cap.release()
PY
