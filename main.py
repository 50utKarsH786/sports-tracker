"""
main.py
-------
CLI entry point for the Sports Multi-Object Tracking pipeline.

Usage:
    python main.py --input video.mp4 --output output/annotated.mp4
    python main.py --input video.mp4 --conf 0.35 --no-speed
    python main.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

from detector    import PersonDetector
from tracker     import SportsTracker
from visualizer  import FrameVisualizer, HeatmapAccumulator


# ══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sports Multi-Object Detection & Persistent ID Tracking"
    )
    p.add_argument("--input",        required=True,          help="Path to input video")
    p.add_argument("--output",       default="output/annotated_video.mp4",
                                                             help="Path to output video")
    p.add_argument("--model",        default="yolov8n.pt",   help="YOLO model weights")
    p.add_argument("--conf",         type=float, default=0.30, help="Detection confidence threshold")
    p.add_argument("--iou",          type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--traj-len",     type=int,   default=60,   help="Trajectory trail length (frames)")
    p.add_argument("--ppm",          type=float, default=20.0, help="Pixels per metre (speed calibration)")
    p.add_argument("--device",       default="cpu",           help="cpu | cuda | mps")
    p.add_argument("--no-traj",      action="store_true",     help="Disable trajectory visualisation")
    p.add_argument("--no-speed",     action="store_true",     help="Disable speed estimation")
    p.add_argument("--heatmap",      action="store_true",     help="Save movement heatmap image")
    p.add_argument("--stats",        default="output/stats.json", help="Path to save JSON stats")
    p.add_argument("--skip-frames",  type=int, default=1,     help="Process every Nth frame (1=all)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
def reenccode_h264(src: str, dst: str) -> None:
    """Re-encode mp4v → h264 for browser-compatible playback."""
    cmd = (
        f'ffmpeg -y -i "{src}" -vcodec libx264 -crf 23 -preset fast '
        f'-movflags +faststart "{dst}" -loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0 or not Path(dst).exists():
        # ffmpeg not available — keep original
        import shutil
        shutil.copy(src, dst)


# ══════════════════════════════════════════════════════════════════════════
def run_pipeline(args: argparse.Namespace) -> None:
    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.input}")

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Pipeline] Input  : {args.input}")
    print(f"[Pipeline] Size   : {width}x{height}  FPS: {fps:.1f}  Frames: {total}")

    # ── Prepare output dir ────────────────────────────────────────────────
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_tmp_raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))

    # ── Initialise modules ────────────────────────────────────────────────
    detector   = PersonDetector(args.model, args.conf, args.iou, args.device)
    tracker    = SportsTracker(fps, args.conf, args.iou, args.traj_len, args.ppm)
    visualizer = FrameVisualizer(
        show_trajectories=not args.no_traj,
        show_speed=not args.no_speed,
    )
    heatmap    = HeatmapAccumulator((height, width)) if args.heatmap else None

    # ── Stats accumulators ────────────────────────────────────────────────
    counts_over_time: list[dict] = []
    frame_idx  = 0
    t0         = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    print("[Pipeline] Processing frames…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed if requested
        if frame_idx % args.skip_frames != 0:
            writer.write(frame)   # write original frame for skipped
            frame_idx += 1
            continue

        # Detect → Track → Visualise
        detections = detector.detect(frame)
        tracked    = tracker.update(detections)

        if heatmap:
            heatmap.add(tracked)

        annotated = visualizer.annotate(frame, tracked, tracker.state, frame_idx)
        writer.write(annotated)

        n = len(tracked)
        counts_over_time.append({"frame": frame_idx, "count": n})

        # Progress print every 60 frames
        if frame_idx % 60 == 0:
            elapsed = time.time() - t0
            pct     = frame_idx / max(total, 1) * 100
            eta     = (elapsed / max(frame_idx, 1)) * (total - frame_idx)
            print(f"  Frame {frame_idx:>5}/{total}  ({pct:5.1f}%)  "
                  f"Subjects: {n:>3}  ETA: {eta:.0f}s")

        frame_idx += 1

    cap.release()
    writer.release()

    # ── Re-encode to h264 ─────────────────────────────────────────────────
    print(f"[Pipeline] Re-encoding to h264 → {args.output}")
    reenccode_h264(tmp_out, args.output)
    Path(tmp_out).unlink(missing_ok=True)

    # ── Save heatmap ──────────────────────────────────────────────────────
    if heatmap:
        hm_path = out_dir / "heatmap.png"
        cv2.imwrite(str(hm_path), heatmap.render())
        print(f"[Pipeline] Heatmap saved → {hm_path}")

    # ── Save stats JSON ───────────────────────────────────────────────────
    stats = {
        "input_video"   : args.input,
        "total_frames"  : frame_idx,
        "fps"           : fps,
        "unique_ids"    : len(tracker.state.unique_ids),
        "all_track_ids" : tracker.state.unique_ids,
        "counts_over_time": counts_over_time,
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[Pipeline] ✅ Done in {elapsed:.1f}s")
    print(f"  Output video : {args.output}")
    print(f"  Unique IDs   : {len(tracker.state.unique_ids)}")
    print(f"  Stats JSON   : {args.stats}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
