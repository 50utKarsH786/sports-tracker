"""
detector.py
-----------
Wraps YOLOv8 for person detection.
Returns supervision Detections objects for easy downstream use.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO
import supervision as sv


class PersonDetector:
    """
    Thin wrapper around YOLOv8 that:
      - loads the model once
      - filters to person class only (class_id = 0)
      - returns sv.Detections for compatibility with ByteTracker
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path:      YOLO weights file. Auto-downloads if not found.
            conf_threshold:  Minimum confidence to keep a detection.
            iou_threshold:   NMS IoU threshold.
            device:          'cpu' or 'cuda' or 'mps'.
        """
        print(f"[Detector] Loading model: {model_path} on {device}")
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device

    # ──────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single BGR frame.

        Args:
            frame: HxWx3 uint8 numpy array (OpenCV BGR).

        Returns:
            sv.Detections with xyxy boxes, confidence scores, class ids.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],          # person only
            verbose=False,
            device=self.device,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        return detections

    # ──────────────────────────────────────────────────────────────────────
    def update_thresholds(self, conf: float, iou: float) -> None:
        """Hot-update thresholds without reloading the model."""
        self.conf_threshold = conf
        self.iou_threshold  = iou
