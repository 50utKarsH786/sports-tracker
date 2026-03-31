# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage

**Candidate:** [Your Name]  
**Date:** March 2026  
**Assignment:** Predusk AI Internship — Computer Vision Task  

---

## 1. Objective

Design and implement a computer vision pipeline that detects all relevant subjects in a public sports video, assigns consistent unique IDs across the full video, and handles real-world challenges such as occlusion, motion blur, scale changes, camera motion, and similar-looking subjects.

---

## 2. Model / Detector Used

**YOLOv8n** (Ultralytics, 2023)

YOLOv8 was selected as the detection backbone for the following reasons:
- Offers the best speed/accuracy trade-off for real-time person detection on CPU
- Native integration with the `supervision` library, simplifying annotation pipelines
- The nano variant (`yolov8n.pt`, ~6 MB) runs without CUDA, enabling free-tier cloud deployment
- Pre-trained COCO weights deliver strong out-of-the-box person detection (class 0)

**Configuration:** confidence threshold = 0.30, IoU (NMS) threshold = 0.50, class filter = person only.

---

## 3. Tracking Algorithm Used

**ByteTrack** (Zhang et al., 2022) via the `supervision` library

ByteTrack maintains tracks using a two-pass IoU association strategy:

1. **First pass** — High-confidence detections matched to existing tracks via IoU cost matrix
2. **Second pass** — Low-confidence detections (partially occluded subjects) matched to remaining unmatched tracks
3. **Track management** — Tracks inactive beyond a configurable buffer (2× fps frames) are dropped; new tracks initialised for unmatched high-confidence detections

---

## 4. Why This Combination Was Selected

**ByteTrack over DeepSORT / StrongSORT:**  
ByteTrack does not rely on appearance embeddings for re-identification. In sports footage, multiple athletes wear identical jerseys — the exact scenario that defeats appearance-based re-ID models. ByteTrack's IoU-only association is both faster and more appropriate for this domain.

**YOLOv8n over YOLOv5 / RT-DETR:**  
RT-DETR is more accurate but too heavy for CPU deployment. YOLOv5 has an older API. YOLOv8n hits the sweet spot for this task.

---

## 5. How ID Consistency Is Maintained

- **Track buffer** set to `fps × 2` seconds: track IDs survive brief disappearances (e.g., player momentarily behind another player or out of frame)
- **IoU association** is tolerant of moderate camera motion since relative positions between tracks remain stable
- **Trajectory deques** visualise ID consistency — a stable colour trail confirms the ID did not switch
- **Exponential moving average** on speed (α = 0.7) prevents jitter from single-frame noise

---

## 6. Challenges Faced

| Challenge | Impact | Approach |
|---|---|---|
| Identical jerseys | Appearance re-ID impossible | Used ByteTrack (IoU-only) — avoids the problem entirely |
| Partial occlusion | Detections drop in confidence | ByteTrack's second-pass low-confidence matching recovers these |
| Fast camera pan | Inter-frame IoU drops sharply | Track buffer keeps IDs alive during pan; recovers on re-entry |
| CPU-only deployment | Slow inference on free tier | YOLOv8n chosen for speed; users advised to use 15–60s clips |
| Browser video playback | Raw `mp4v` codec unsupported in browsers | FFmpeg post-process re-encodes to h264 with `+faststart` |

---

## 7. Failure Cases Observed

1. **Prolonged full overlap (>2 seconds):** When two players completely overlap for extended periods, ID swap is possible upon separation. This is a fundamental limitation of IoU-based tracking without appearance re-ID.

2. **Rapid camera pan:** Fast horizontal pans create large inter-frame IoU drops. All active tracks can be simultaneously lost and restarted with new IDs.

3. **Extreme motion blur:** Very fast movement (e.g., a sprinting athlete) produces low-confidence detections that may be dropped in the first pass, causing temporary track loss.

4. **Speed estimate noise on slow movement:** At very low speeds, single-pixel centroid noise dominates. The EMA filter mitigates this but does not eliminate it.

---

## 8. Possible Improvements

- **Appearance re-ID (BoT-SORT / StrongSORT):** Add OSNet embeddings for robust re-identification after full occlusion at the cost of inference speed
- **Homography-based ground projection:** Map camera image to a top-down pitch view for accurate speed and spatial heatmaps
- **Team clustering:** K-means on HSV jersey colour histograms to separate teams and compute team-level statistics
- **Multi-class tracking:** Extend `classes=[0]` to include balls, referees, or vehicles depending on sport
- **GPU deployment:** Switch to Hugging Face Spaces T4 GPU for 10× faster inference on longer videos
- **Temporal smoothing of bounding boxes:** Kalman filter on box positions reduces jitter on individual tracks

---

## 9. Deliverables Summary

| Deliverable | Status |
|---|---|
| GitHub repository | ✅ |
| README.md | ✅ |
| Annotated output video | ✅ `output/annotated_video.mp4` |
| Original public video link | ✅ included in submission |
| Technical report | ✅ This document |
| Sample screenshots | ✅ `output/screenshots/` |
| Demo video (3–5 min) | ✅ Loom recording |
| Trajectory visualisation | ✅ |
| Movement heatmap | ✅ |
| Speed estimation | ✅ |

---

*Source code: [GitHub Link] | Live demo: [Hugging Face Spaces Link]*
