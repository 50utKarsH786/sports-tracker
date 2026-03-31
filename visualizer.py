"""
visualizer.py
-------------
All frame annotation logic:
  - Bounding boxes with unique colours per ID
  - ID + speed labels
  - Fading trajectory trails
  - HUD overlay (subject count, frame number)
  - Heatmap accumulation (optional)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from tracker import TrackState


# ── 20-colour palette (visually distinct) ─────────────────────────────────
PALETTE: List[Tuple[int, int, int]] = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  199),
]


def color_for_id(tid: int) -> Tuple[int, int, int]:
    """Deterministic colour for a given track ID."""
    return PALETTE[int(tid) % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════
class FrameVisualizer:
    """
    Stateless annotator — takes a frame + detections + TrackState,
    returns an annotated copy.
    """

    def __init__(
        self,
        show_trajectories: bool = True,
        show_speed: bool        = True,
        box_thickness: int      = 2,
        label_scale: float      = 0.55,
    ) -> None:
        self.show_trajectories = show_trajectories
        self.show_speed        = show_speed

        self.box_annotator = sv.BoxAnnotator(thickness=box_thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=label_scale,
            text_thickness=1,
            text_padding=4,
        )

    # ──────────────────────────────────────────────────────────────────────
    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        state: TrackState,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Full annotation pipeline for one frame.

        Args:
            frame:      Original BGR frame (not mutated).
            detections: Tracked sv.Detections with tracker_id.
            state:      TrackState holding trajectories + speeds.
            frame_idx:  Current frame number (for HUD).

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()

        # 1. Draw trajectory trails ────────────────────────────────────────
        if self.show_trajectories:
            out = self._draw_trajectories(out, state)

        # 2. Bounding boxes ───────────────────────────────────────────────
        out = self.box_annotator.annotate(scene=out, detections=detections)

        # 3. Labels (ID + speed) ──────────────────────────────────────────
        labels = self._build_labels(detections, state)
        out = self.label_annotator.annotate(
            scene=out, detections=detections, labels=labels
        )

        # 4. HUD overlay ──────────────────────────────────────────────────
        out = self._draw_hud(out, detections, frame_idx)

        return out

    # ──────────────────────────────────────────────────────────────────────
    def _draw_trajectories(
        self, frame: np.ndarray, state: TrackState
    ) -> np.ndarray:
        overlay = frame.copy()

        for tid, pts_deque in state.trajectories.items():
            pts = list(pts_deque)
            if len(pts) < 2:
                continue
            col = color_for_id(tid)
            n   = len(pts)
            for j in range(1, n):
                alpha = j / n                          # older = dimmer
                c = tuple(int(v * alpha) for v in col)
                cv2.line(overlay, pts[j - 1], pts[j], c, 2, cv2.LINE_AA)

        # Blend trail with original frame for glow effect
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        return frame

    # ──────────────────────────────────────────────────────────────────────
    def _build_labels(
        self, detections: sv.Detections, state: TrackState
    ) -> List[str]:
        labels = []
        for tid in (detections.tracker_id or []):
            if tid is None:
                labels.append("?")
                continue
            tid = int(tid)
            spd = state.speed(tid)
            speed_str = f"  {spd:.1f} km/h" if (self.show_speed and spd is not None) else ""
            labels.append(f"#{tid}{speed_str}")
        return labels

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        detections: sv.Detections,
        frame_idx: int,
    ) -> np.ndarray:
        n   = len(detections)
        txt = f"Subjects: {n}   Frame: {frame_idx}"

        # Dark pill background
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8, 8), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(
            frame, txt, (14, th + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 187), 1, cv2.LINE_AA
        )
        return frame


# ══════════════════════════════════════════════════════════════════════════
class HeatmapAccumulator:
    """
    Accumulates subject positions across all frames.
    Call .add(frame, detections) each frame, then .render() at the end.
    """

    def __init__(self, frame_shape: Tuple[int, int]) -> None:
        h, w = frame_shape
        self.heatmap = np.zeros((h, w), dtype=np.float32)

    def add(self, detections: sv.Detections) -> None:
        if detections.tracker_id is None:
            return
        for xyxy in detections.xyxy:
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
                cv2.circle(self.heatmap, (cx, cy), 15, 1.0, -1)

    def render(self) -> np.ndarray:
        """Returns a colourised BGR heatmap image."""
        norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return coloured
