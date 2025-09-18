# src/events.py
from typing import Dict, List, Tuple
import math

# keep short centroid history per ID to estimate speed
_hist: Dict[int, List[Tuple[float,float]]] = {}

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / max(1, area_a + area_b - inter)

def update_and_check(tracks: List[Tuple], fps: float = 15.0):
    """Return list of incidents with a simple severity."""
    incidents = []
    # update histories
    for (tid,x1,y1,x2,y2,cls,conf,cx,cy) in tracks:
        _hist.setdefault(tid, []).append((cx,cy))
        _hist[tid] = _hist[tid][-10:]

    # Sudden-stop rule: recent speed << previous average
    for tid, pts in _hist.items():
        if len(pts) >= 5:
            d = [math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1,len(pts))]
            if len(d) >= 4:
                prev = max(1e-3, sum(d[:-2])/(len(d)-2) * fps)
                recent = (d[-1]+d[-2])/2 * fps
                if prev > 30 and recent < prev*0.4:
                    incidents.append({"type":"sudden_stop","track_id":tid,"severity":"Minor"})

    # Overlap rule: high IoU between any two vehicles
    for i in range(len(tracks)):
        for j in range(i+1, len(tracks)):
            a = tracks[i]; b = tracks[j]
            if _iou((a[1],a[2],a[3],a[4]), (b[1],b[2],b[3],b[4])) > 0.25:
                incidents.append({"type":"overlap","a":a[0],"b":b[0],"severity":"Moderate"})
    return incidents
