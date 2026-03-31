"""
tracker.py
----------
Wraps supervision's ByteTracker and maintains per-ID state:
  - trajectory history (deque of centroids)
  - speed estimates (km/h)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


# ── Rough calibration: pixels-per-metre ───────────────────────────────────
# Assumes ~200 px visible width ≈ 10 m of real-world pitch.
# Override via TrackState.ppm for better accuracy.
DEFAULT_PPM = 20.0   # pixels per metre


class TrackState:
    """Holds all mutable state for active/past tracks."""

    def __init__(self, trajectory_maxlen: int = 60, ppm: float = DEFAULT_PPM) -> None:
        self.trajectory_maxlen = trajectory_maxlen
        self.ppm               = ppm

        # track_id  →  deque of (cx, cy) pixel centroids
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trajectory_maxlen)
        )
        # track_id  →  last centroid (for speed delta)
        self.prev_centers: Dict[int, Tuple[int, int]] = {}
        # track_id  →  smoothed speed in km/h
        self.speeds: Dict[int, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    def update(self, detections: sv.Detections, fps: float) -> None:
        """
        Called once per frame after tracking.
        Updates trajectories and speed estimates for all tracked detections.

        Args:
            detections: sv.Detections with .tracker_id populated.
            fps:        Video frame rate (for speed calculation).
        """
        if detections.tracker_id is None:
            return

        new_centers: Dict[int, Tuple[int, int]] = {}

        for i, tid in enumerate(detections.tracker_id):
            if tid is None:
                continue
            tid = int(tid)
            xyxy = detections.xyxy[i]
            cx   = int((xyxy[0] + xyxy[2]) / 2)
            cy   = int((xyxy[1] + xyxy[3]) / 2)

            new_centers[tid] = (cx, cy)
            self.trajectories[tid].append((cx, cy))

            # Speed: frame-to-frame euclidean distance → km/h
            if tid in self.prev_centers and fps > 0:
                px0, py0  = self.prev_centers[tid]
                dist_px   = math.hypot(cx - px0, cy - py0)
                dist_m    = dist_px / self.ppm
                speed_ms  = dist_m * fps
                speed_kmh = speed_ms * 3.6

                # Exponential moving average to smooth jitter
                prev_speed = self.speeds.get(tid, speed_kmh)
                self.speeds[tid] = 0.7 * prev_speed + 0.3 * speed_kmh

        self.prev_centers = new_centers

    # ──────────────────────────────────────────────────────────────────────
    def trajectory(self, tid: int) -> List[Tuple[int, int]]:
        return list(self.trajectories[tid])

    def speed(self, tid: int) -> Optional[float]:
        return self.speeds.get(tid)

    @property
    def unique_ids(self) -> List[int]:
        return list(self.trajectories.keys())


# ══════════════════════════════════════════════════════════════════════════
class SportsTracker:
    """
    Combines supervision ByteTracker with TrackState.
    Single call per frame: tracker.update(detections, fps) → sv.Detections
    """

    def __init__(
        self,
        fps: float = 30.0,
        conf_threshold: float = 0.30,
        iou_threshold: float  = 0.50,
        trajectory_len: int   = 60,
        ppm: float            = DEFAULT_PPM,
    ) -> None:
        self.fps = fps

        self.byte_tracker = sv.ByteTracker(
            track_activation_threshold=conf_threshold,
            lost_track_buffer=int(fps * 2),      # keep ID alive 2 sec
            minimum_matching_threshold=iou_threshold,
            frame_rate=int(fps),
        )

        self.state = TrackState(trajectory_maxlen=trajectory_len, ppm=ppm)

    # ──────────────────────────────────────────────────────────────────────
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Args:
            detections: Raw sv.Detections from PersonDetector.

        Returns:
            sv.Detections with .tracker_id assigned.
        """
        tracked = self.byte_tracker.update_with_detections(detections)
        self.state.update(tracked, self.fps)
        return tracked
