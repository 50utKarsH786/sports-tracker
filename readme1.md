---
title: Sports Multi-Object Tracker
emoji: 🏃
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
short_description: YOLOv8 + ByteTrack — detect & track players in sports videos
---

#  Sports Multi-Object Tracker

> **YOLOv8n + ByteTrack** — Detect, track, and annotate all subjects in sports/event videos with persistent unique IDs, trajectory trails, speed estimation, and movement heatmaps.

---

## Project Structure

```
sports-tracker/
├── app.py              ← Gradio web app (HF Spaces entry point)
├── main.py             ← CLI pipeline runner
├── detector.py         ← YOLOv8 detection module
├── tracker.py          ← ByteTrack wrapper + track state
├── visualizer.py       ← Bounding boxes, labels, trails, heatmap
├── notebook.ipynb      ← Step-by-step walkthrough
├── requirements.txt    ← All dependencies
├── output/
│   ├── annotated_video.mp4
│   └── screenshots/
└── README.md           ← This file
```

---

##  Quick Start — Local

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sports-tracker.git
cd sports-tracker

# 2. Install
pip install -r requirements.txt

# 3a. Run web UI
python app.py
# → http://localhost:7860

# 3b. OR run CLI
python main.py --input your_video.mp4 --output output/annotated_video.mp4

# 3c. OR open notebook
jupyter notebook notebook.ipynb
```

YOLOv8n weights (~6 MB) download automatically on first run.

---

## ☁️ Deploy to Hugging Face Spaces

```bash
# 1. Create Space at https://huggingface.co/new-space
#    SDK: Gradio | Hardware: CPU Basic (free)

# 2. Push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/sports-tracker
git push hf main

# HF auto-installs requirements.txt and launches app.py
```

---

## ⚙️ CLI Options

```
python main.py --input VIDEO --output OUTPUT [options]

  --conf       0.30    Detection confidence threshold
  --iou        0.50    NMS IoU threshold
  --traj-len   60      Trajectory trail length in frames
  --ppm        20.0    Pixels per metre (speed calibration)
  --device     cpu     cpu | cuda | mps
  --no-traj            Disable trajectory trails
  --no-speed           Disable speed estimates
  --heatmap            Save movement heatmap image
  --skip-frames 1      Process every Nth frame (1 = all)
  --stats      output/stats.json   Path to save JSON stats
```

---

## 🧰 Tech Stack

| Component | Library | Why |
|---|---|---|
| Detection | YOLOv8n (Ultralytics) | Best speed/accuracy on CPU, COCO pre-trained |
| Tracking | ByteTrack (supervision) | No appearance embeddings → robust for identical jerseys |
| Video I/O | OpenCV | Industry standard |
| Annotation | supervision | Clean API for boxes, labels, colours |
| Web UI | Gradio | One-click HF Spaces deployment |

---

##  Assumptions & Limitations

- **Person class only** — change `classes=[0]` in `detector.py` to add other classes
- **Speed is approximate** — uses fixed `PPM=20 px/m` calibration; real accuracy needs camera homography
- **CPU deployment** — ~1–3× real-time on HF free tier; GPU Space = 10× faster
- **ID switches** — can occur during prolonged full occlusion; inherent to IoU-only tracking
- **Recommended clip length** — 15–90 seconds on HF CPU free tier

---

##  Technical Choices

**Why ByteTrack over DeepSORT?**
ByteTrack does not rely on appearance embeddings for re-identification. In sports, athletes wear identical jerseys — appearance-based re-ID fails completely. ByteTrack's two-pass IoU association handles this robustly.

**Why YOLOv8n?**
Fastest YOLO variant with strong person detection. Nano weights (~6 MB) load instantly and run on CPU without CUDA.

---

## 📄 License

MIT
