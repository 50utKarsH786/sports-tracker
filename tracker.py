"""
tracker.py
ByteTrack wrapper using supervision 0.21 stable API.
Maintains trajectory history and speed estimates per track ID.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# pixels-per-metre calibration (rough: 200px ≈ 10m pitch width)
DEFAULT_PPM = 20.0


class TrackState:
    """Stores trajectory + speed per track ID."""

    def __init__(self, traj_len: int = 60, ppm: float = DEFAULT_PPM) -> None:
        self.ppm      = ppm
        self.traj_len = traj_len
        self.trajs:  Dict[int, deque]          = defaultdict(lambda: deque(maxlen=traj_len))
        self.prev:   Dict[int, Tuple[int,int]] = {}
        self.speeds: Dict[int, float]          = {}

    def update(self, tracks: list[dict], fps: float) -> None:
        """
        tracks: list of {"id": int, "xyxy": [x1,y1,x2,y2]}
        """
        new_prev: Dict[int, Tuple[int,int]] = {}
        for t in tracks:
            tid = int(t["id"])
            x1, y1, x2, y2 = t["xyxy"]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            new_prev[tid] = (cx, cy)
            self.trajs[tid].append((cx, cy))

            if tid in self.prev and fps > 0:
                d   = math.hypot(cx - self.prev[tid][0], cy - self.prev[tid][1])
                spd = (d / self.ppm) * fps * 3.6          # km/h
                old = self.speeds.get(tid, spd)
                self.speeds[tid] = 0.7 * old + 0.3 * spd  # EMA smooth

        self.prev = new_prev

    def trajectory(self, tid: int) -> List[Tuple[int,int]]:
        return list(self.trajs[tid])

    def speed(self, tid: int) -> Optional[float]:
        return self.speeds.get(tid)

    @property
    def all_ids(self) -> List[int]:
        return list(self.trajs.keys())


class SportsTracker:
    """
    Wraps supervision 0.21 ByteTracker.
    Input/output uses plain Python dicts — no supervision objects exposed.
    """

    def __init__(
        self,
        fps: float      = 30.0,
        conf: float     = 0.30,
        iou: float      = 0.50,
        traj_len: int   = 60,
        ppm: float      = DEFAULT_PPM,
    ) -> None:
        import supervision as sv

        self.fps   = fps
        self.state = TrackState(traj_len=traj_len, ppm=ppm)

        # supervision 0.21 ByteTrack constructor
        self._tracker = sv.ByteTrack(
            track_activation_threshold=conf,
            lost_track_buffer=max(30, int(fps * 2)),
            minimum_matching_threshold=iou,
            frame_rate=int(fps),
        )

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Args:
            detections: list of {"xyxy": [x1,y1,x2,y2], "conf": float}

        Returns:
            list of {"id": int, "xyxy": [x1,y1,x2,y2], "conf": float}
        """
        import supervision as sv

        if not detections:
            return []

        xyxy  = np.array([d["xyxy"]  for d in detections], dtype=np.float32)
        confs = np.array([d["conf"]  for d in detections], dtype=np.float32)
        cids  = np.zeros(len(detections), dtype=int)

        sv_det = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=cids,
        )

        tracked = self._tracker.update_with_detections(sv_det)

        results = []
        if tracked.tracker_id is None:
            return results

        for i, tid in enumerate(tracked.tracker_id):
            if tid is None:
                continue
            results.append({
                "id":   int(tid),
                "xyxy": tracked.xyxy[i].tolist(),
                "conf": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
            })

        self.state.update(results, self.fps)
        return results
