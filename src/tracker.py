# src/tracker.py
from typing import Dict, List, Tuple
import math

# next track id + last known centroid per id
_next_id = 1
_tracks: Dict[int, Tuple[float, float]] = {}  # id -> (cx, cy)

def track(
    boxes: List[Tuple[int,int,int,int,int,float]],
    max_dist: float = 60.0
) -> List[Tuple[int,int,int,int,int,float,float,float]]:
    """
    Assigns an ID to each detection by nearest centroid.
    Input  boxes: (x1,y1,x2,y2,cls,conf)
    Output list : (id,x1,y1,x2,y2,cls,conf,cx,cy)
    """
    global _next_id, _tracks
    out = []
    new_tracks: Dict[int, Tuple[float,float]] = {}

    for (x1, y1, x2, y2, cls, conf) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # find nearest existing track
        best_id, best_d = None, 1e9
        for tid, (px, py) in _tracks.items():
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_id, best_d = tid, d

        if best_id is not None and best_d <= max_dist:
            tid = best_id
        else:
            tid = _next_id
            _next_id += 1

        new_tracks[tid] = (cx, cy)
        out.append((tid, x1, y1, x2, y2, cls, conf, cx, cy))

    _tracks = new_tracks
    return out
