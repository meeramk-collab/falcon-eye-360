# src/detector.py
from typing import List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics not installed. Run: pip install ultralytics"
    ) from e

# load a tiny, fast model once
_model = YOLO("yolov8n.pt")  # downloads on first use

# COCO ids to keep: person,bicycle,car,motorbike,bus,truck
VEHICLE_PEDESTRIAN_CLASSES = {0, 1, 2, 3, 5, 7}

def detect(frame_bgr: np.ndarray, conf: float = 0.35) -> List[Tuple[int,int,int,int,int,float]]:
    """
    Run YOLO on a single BGR frame (OpenCV format).
    Returns a list of (x1, y1, x2, y2, cls_id, confidence).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    # predict on numpy array; verbose=False to keep console clean
    res = _model.predict(frame_bgr, conf=conf, verbose=False)[0]
    out = []
    for b in res.boxes:
        cls = int(b.cls.item())
        if cls not in VEHICLE_PEDESTRIAN_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        confv = float(b.conf.item())
        out.append((x1, y1, x2, y2, cls, confv))
    return out

def draw_boxes(frame_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int,int,float]]) -> np.ndarray:
    """Draw simple rectangles + conf on the frame (OpenCV)."""
    import cv2
    for (x1, y1, x2, y2, cls, conf) in boxes:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"{cls}:{conf:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame_bgr
