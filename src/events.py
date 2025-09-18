# src/events.py
from typing import Dict, List, Tuple
import math

# keep the last few centroids per track id
_hist: Dict[int, List[Tuple[float, float]]] = {}

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)

def update_and_check(
    tracked: List[Tuple[int,int,int,int,int,float,float,float]],
    fps: float = 15.0,
    stop_drop_ratio: float = 0.4,  # recent speed < 40% of previous avg
    min_prev_speed: float = 30.0,  # pixels/sec baseline
    overlap_iou: float = 0.25
) -> List[dict]:
    """
    Input  tracked: (id,x1,y1,x2,y2,cls,conf,cx,cy)
    Output incidents: dicts like {"type": "...", "severity": "...", ...}
    """
    incidents: List[dict] = []

    # update histories
    for (tid, x1, y1, x2, y2, cls, conf, cx, cy) in tracked:
        _hist.setdefault(tid, []).append((cx, cy))
        _hist[tid] = _hist[tid][-10:]  # last ~10 positions

    # Sudden stop: recent speed << previous average
    for tid, pts in _hist.items():
        if len(pts) >= 5:
            d = [
                math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                for i in range(1, len(pts))
            ]
            if len(d) >= 4:
                prev = max(1e-3, sum(d[:-2]) / max(1, len(d) - 2)) * fps
                recent = ((d[-1] + d[-2]) / 2.0) * fps
                if prev > min_prev_speed and recent < prev * stop_drop_ratio:
                    incidents.append({
                        "type": "sudden_stop",
                        "track_id": tid,
                        "severity": "Minor"
                    })

    # Overlap: high IoU between two tracked vehicles
    for i in range(len(tracked)):
        for j in range(i + 1, len(tracked)):
            a = tracked[i]; b = tracked[j]
            box_a = (a[1], a[2], a[3], a[4])
            box_b = (b[1], b[2], b[3], b[4])
            if _iou(box_a, box_b) > overlap_iou:
                incidents.append({
                    "type": "overlap",
                    "a": a[0],
                    "b": b[0],
                    "severity": "Moderate"
                })

    return incidents
