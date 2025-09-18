# src/tracker.py
from typing import Dict, List, Tuple
import math
_next_id = 1
_tracks: Dict[int, Tuple[float,float]] = {}  # id -> (cx, cy)

def track(boxes: List[Tuple[int,int,int,int,int,float]], max_dist: float = 60):
    """Assign an ID to each detection by nearest centroid."""
    global _next_id, _tracks
    assigned = []
    new_tracks = {}
    for (x1,y1,x2,y2,cls,conf) in boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        best_id, best_d = None, 1e9
        for tid,(px,py) in _tracks.items():
            d = math.hypot(cx-px, cy-py)
            if d < best_d:
                best_id, best_d = tid, d
        if best_d <= max_dist and best_id is not None:
            tid = best_id
        else:
            tid = _next_id; _next_id += 1
        new_tracks[tid] = (cx,cy)
        assigned.append((tid,x1,y1,x2,y2,cls,conf,cx,cy))
    _tracks = new_tracks
    return assigned  # (id,x1,y1,x2,y2,cls,conf,cx,cy)
