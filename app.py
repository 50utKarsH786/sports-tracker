"""
app.py  —  Sports Observer
Gradio app for Hugging Face Spaces.
UI follows DESIGN.md "The Digital Observer" spec exactly:
  background   #0b0e14   void black
  surface      #161a21   primary workspace
  surface-high #1c2028   panels
  primary      #a1ffc2   green accent
  secondary    #00d2fd   cyan accent
  tertiary     #ff7350   orange alert
  on-surface   #ecedf6   body text (never pure white)
  Space Grotesk headlines / Inter body / IBM Plex Mono data
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import traceback
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

# ── Palette (BGR for OpenCV) ───────────────────────────────────────────────
PALETTE_BGR = [
    (253,210,0),(194,255,161),(80,115,255),(187,212,0),
    (29,178,255),(134,219,61),(56,56,255),(255,115,100),
    (255,194,0),(49,210,207),(151,157,255),(23,204,146),
    (255,56,132),(31,112,255),(52,147,26),(255,56,203),
    (168,153,44),(200,149,255),(10,249,72),(133,0,82),
]
PPM = 20.0   # pixels per metre


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def process_video(video_path, conf, iou, show_traj, show_speed, traj_len, progress=gr.Progress()):
    if video_path is None:
        return None, None, '{"status":"waiting"}', _status("idle", "Upload a video to begin.")
    try:
        return _run(video_path, float(conf), float(iou), bool(show_traj),
                    bool(show_speed), int(traj_len), progress)
    except Exception as exc:
        traceback.print_exc()
        return None, None, json.dumps({"error": str(exc)}), _status("error", str(exc))


def _find_working_fourcc(fps, W, H):
    """Try several codecs and return (fourcc, suffix) for the first one that works."""
    import shutil
    candidates = [
        ("avc1", ".mp4"),   # H.264 — best for browsers
        ("H264", ".mp4"),
        ("X264", ".mp4"),
        ("mp4v", ".mp4"),   # MPEG-4 fallback  (needs re-encode for browser)
    ]
    for codec, ext in candidates:
        test_path = tempfile.mktemp(suffix=f"_test{ext}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(test_path, fourcc, fps, (W, H))
            if w.isOpened():
                # Write a test frame to make sure it really works
                w.write(np.zeros((H, W, 3), dtype=np.uint8))
                w.release()
                if Path(test_path).exists() and Path(test_path).stat().st_size > 0:
                    Path(test_path).unlink(missing_ok=True)
                    print(f"[Codec] Using {codec}")
                    return fourcc, ext, codec
            w.release()
        except Exception:
            pass
        finally:
            Path(test_path).unlink(missing_ok=True)
    # absolute fallback
    print("[Codec] Falling back to mp4v")
    return cv2.VideoWriter_fourcc(*"mp4v"), ".mp4", "mp4v"


def _reencode_for_browser(input_path, output_path):
    """Try to re-encode to H.264 with ffmpeg/ffmpeg.exe. Returns True on success."""
    import subprocess, shutil

    # Check if ffmpeg is available
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd is None:
        print("[Encode] ffmpeg not found, skipping re-encode")
        return False

    try:
        result = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", input_path,
             "-vcodec", "libx264", "-crf", "23",
             "-preset", "fast", "-movflags", "+faststart",
             output_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            print("[Encode] H.264 re-encode successful")
            return True
        else:
            print(f"[Encode] ffmpeg failed: {result.stderr[:300]}")
            return False
    except Exception as e:
        print(f"[Encode] ffmpeg error: {e}")
        return False


def _run(video_path, conf, iou, show_traj, show_speed, traj_len, progress):
    from ultralytics import YOLO
    import supervision as sv

    # ── open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # ── writers ───────────────────────────────────────────────────────────
    fourcc, ext, codec_name = _find_working_fourcc(fps, W, H)
    tmp_raw = tempfile.mktemp(suffix=f"_raw{ext}")
    tmp_out = tempfile.mktemp(suffix="_out.mp4")
    writer  = cv2.VideoWriter(tmp_raw, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer with codec {codec_name}. "
                           "Please install ffmpeg or an H.264-capable OpenCV build.")

    # ── model ─────────────────────────────────────────────────────────────
    model = YOLO("yolov8n.pt")

    # ── tracker (supervision 0.21 stable API) ─────────────────────────────
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=conf,
        lost_track_buffer=max(30, int(fps * 2)),
        minimum_matching_threshold=iou,
        frame_rate=int(fps),
    )

    # ── state ─────────────────────────────────────────────────────────────
    trajs: dict = defaultdict(lambda: deque(maxlen=traj_len))
    prev_c: dict = {}
    speeds: dict = {}
    hm_acc = np.zeros((H, W), dtype=np.float32)
    counts = []
    fi     = 0

    progress(0, desc="Initialising…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect
        res  = model(frame, conf=conf, iou=iou, classes=[0], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)

        # track
        if len(dets) > 0:
            tracked = byte_tracker.update_with_detections(dets)
        else:
            tracked = sv.Detections.empty()

        out = frame.copy()

        # collect active tracks
        active_tracks = []
        if tracked.tracker_id is not None and len(tracked) > 0:
            for i, tid in enumerate(tracked.tracker_id):
                if tid is None:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
                active_tracks.append({"id": tid, "box": (x1,y1,x2,y2)})

                cx, cy = (x1+x2)//2, (y1+y2)//2
                trajs[tid].append((cx, cy))
                if 0 <= cy < H and 0 <= cx < W:
                    cv2.circle(hm_acc, (cx, cy), 18, 1.0, -1)

                # speed EMA
                if show_speed and tid in prev_c:
                    d   = math.hypot(cx - prev_c[tid][0], cy - prev_c[tid][1])
                    spd = (d / PPM) * fps * 3.6
                    speeds[tid] = 0.7 * speeds.get(tid, spd) + 0.3 * spd
                prev_c[tid] = (cx, cy)

        # ── draw trajectories ─────────────────────────────────────────────
        if show_traj:
            ovl = out.copy()
            for tid, pts_dq in trajs.items():
                pts = list(pts_dq)
                col = PALETTE_BGR[tid % len(PALETTE_BGR)]
                for j in range(1, len(pts)):
                    a = j / max(len(pts), 1)
                    c = tuple(int(v * a) for v in col)
                    cv2.line(ovl, pts[j-1], pts[j], c, 2, cv2.LINE_AA)
            cv2.addWeighted(ovl, 0.70, out, 0.30, 0, out)

        # ── draw boxes + labels (DESIGN.md colours) ───────────────────────
        for t in active_tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["box"]

            # secondary #00d2fd (BGR: 253,210,0)
            cv2.rectangle(out, (x1,y1), (x2,y2), (253,210,0), 2)

            s_str = f" {speeds[tid]:.0f}km/h" if (show_speed and tid in speeds) else ""
            label = f"#{tid}{s_str}"
            fs, tk = 0.45, 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tk)
            lx = x1
            ly = max(y1 - 4, th + 6)
            # secondary-container #00677e (BGR: 126,103,0)
            cv2.rectangle(out, (lx, ly-th-4), (lx+tw+8, ly+2), (126,103,0), -1)
            # on-secondary-container #eefaff
            cv2.putText(out, label, (lx+4, ly-1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255,250,238), tk, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────
        n   = len(active_tracks)
        hud = f"SUBJECTS:{n:02d}  FRAME:{fi:05d}"
        (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
        cv2.rectangle(out, (8,8), (hw+20, hh+16), (0,0,0), -1)
        # primary #a1ffc2 (BGR: 194,255,161)
        cv2.putText(out, hud, (13, hh+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (194,255,161), 1, cv2.LINE_AA)

        writer.write(out)
        counts.append({"frame": fi, "count": n})
        fi += 1

        if fi % 25 == 0:
            progress(fi / total, desc=f"Frame {fi}/{total}  ·  Subjects: {n}")

    cap.release()
    writer.release()

    # ── re-encode to browser-compatible H.264 if needed ───────────────────
    final = tmp_raw
    if codec_name not in ("avc1", "H264", "X264"):
        # mp4v isn't browser-playable, try re-encoding with ffmpeg
        if _reencode_for_browser(tmp_raw, tmp_out):
            final = tmp_out
            Path(tmp_raw).unlink(missing_ok=True)
        else:
            # Last resort: serve the mp4v file as-is; Gradio may still handle it
            print("[Warning] Output video may not play in browser without ffmpeg. "
                  "Install ffmpeg for best results: https://ffmpeg.org/download.html")
            final = tmp_raw

    # ── heatmap ───────────────────────────────────────────────────────────
    hm_path = tempfile.mktemp(suffix="_hm.png")
    norm = cv2.normalize(hm_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(hm_path, cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    uid   = len(trajs)
    stats = json.dumps({
        "total_frames"    : fi,
        "unique_ids"      : uid,
        "all_track_ids"   : list(trajs.keys()),
        "fps"             : round(fps, 2),
        "counts_over_time": counts[-300:],
    }, indent=2)

    return (
        final,
        hm_path,
        stats,
        _status("ok", f"Complete  ·  {fi} frames processed  ·  {uid} unique IDs tracked"),
    )


def _status(kind: str, msg: str) -> str:
    cfg = {
        "ok":    ("#a1ffc2", "SYSTEM NOMINAL"),
        "error": ("#ff716c", "SYSTEM ERROR"),
        "idle":  ("#45484f", "STANDBY"),
    }
    col, prefix = cfg.get(kind, cfg["idle"])
    dot_anim = "animation:pulse 2s ease-in-out infinite;" if kind == "ok" else ""
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;
            background:{col}14;border-radius:6px;margin-top:8px;">
  <span style="width:7px;height:7px;border-radius:50%;background:{col};
               flex-shrink:0;{dot_anim}"></span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
               color:{col};letter-spacing:.06em;">
    <span style="opacity:.5;margin-right:8px;">{prefix}</span>{msg}
  </span>
</div>
<style>
  @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.7)}}}}
</style>"""


# ══════════════════════════════════════════════════════════════════════════
#  DESIGN.md CSS  —  "The Digital Observer"
# ══════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0b0e14 !important; }

.gradio-container {
    background: #0b0e14 !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 2.25rem !important;
    font-family: 'Inter', sans-serif !important;
    color: #ecedf6 !important;
}

/* ── Masthead ── */
#masthead {
    background: #161a21;
    border-radius: 10px;
    padding: 1.75rem 2rem 1.5rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
/* sensor-sweep gradient texture (DESIGN.md §2) */
#masthead::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, #a1ffc21a 0%, #00fc9a0d 40%, transparent 70%);
    pointer-events: none;
}
#masthead .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .65rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #3d4555;
    margin-bottom: .5rem;
    display: block;
}
#masthead h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ecedf6;
    letter-spacing: -.03em;
    line-height: 1.1;
}
#masthead h1 em {
    font-style: normal;
    color: #a1ffc2;
}
.badge-row {
    display: flex;
    gap: 6px;
    margin-top: .85rem;
    flex-wrap: wrap;
}
.bdg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .6rem;
    letter-spacing: .07em;
    padding: 3px 9px;
    border-radius: 4px;
    border: 1px solid;
}
.bdg-p { color:#a1ffc2; border-color:#a1ffc228; background:#a1ffc20e; }
.bdg-s { color:#00d2fd; border-color:#00d2fd28; background:#00d2fd0e; }
.bdg-t { color:#ff7350; border-color:#ff735028; background:#ff73500e; }
.bdg-n { color:#45484f; border-color:#45484f40; }

/* ── Workspace grid ── */
.workspace {
    display: grid;
    grid-template-columns: 310px 1fr;
    gap: 1.75rem;
    align-items: start;
}

/* ── Control panel ── */
.ctrl {
    background: #161a21;
    border-radius: 10px;
    padding: .9rem;
}
/* Section labels — no borders, tonal only (DESIGN.md No-Line rule) */
.sec {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .6rem;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #2e3340;
    padding: .75rem 0 .3rem;
    margin-top: .5rem;
}
.sec:first-child { padding-top: 0; margin-top: 0; }

/* ── Gradio element overrides ── */
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .svelte-1gfkn6j {
    font-family: 'Inter', sans-serif !important;
    font-size: .78rem !important;
    color: #6b7585 !important;
    font-weight: 400 !important;
}
.gradio-container input[type=range] { accent-color: #00d2fd !important; }
.gradio-container input[type=checkbox] { accent-color: #a1ffc2 !important; }
.gradio-container .wrap { background: #161a21 !important; border: none !important; }

/* ── Primary CTA (DESIGN.md §5 Buttons) ── */
#run-btn > button {
    width: 100% !important;
    background: #a1ffc2 !important;
    color: #00391e !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;   /* rounded-md */
    height: 44px !important;
    margin-top: .9rem !important;
    letter-spacing: .02em !important;
    transition: opacity .15s, transform .1s !important;
    cursor: pointer !important;
}
#run-btn > button:hover  { opacity: .86 !important; }
#run-btn > button:active { transform: scale(.98) !important; }

/* ── Output panel ── */
.out-panel {
    display: flex;
    flex-direction: column;
    gap: .9rem;
}

/* ── Telemetry cards — glassmorphism (DESIGN.md §2 Glass Rule) ── */
.telem {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.tcard {
    background: rgba(34, 38, 47, .60);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 8px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}
/* sensor-sweep top accent */
.tcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #a1ffc2, #00fc9a);
    opacity: .10;
}
.tv {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.tk {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .58rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #2e3340;
}
.ca { color: #a1ffc2; }   /* primary   */
.cs { color: #00d2fd; }   /* secondary */
.ct { color: #ff7350; }   /* tertiary  */

/* ── Video well — recessed (DESIGN.md §4 Layering) ── */
.video-well {
    background: #000000;
    border-radius: 8px;
    overflow: hidden;
}
.gradio-container video { border-radius: 6px; background: #000; }

/* ── Tabs ── */
.gradio-container .tab-nav {
    background: #10131a !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
    padding: 0 6px !important;
}
.gradio-container .tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .68rem !important;
    letter-spacing: .07em !important;
    color: #2e3340 !important;
    border: none !important;
    padding: 9px 16px !important;
    background: transparent !important;
    text-transform: uppercase !important;
}
.gradio-container .tab-nav button.selected {
    color: #00d2fd !important;
    border-bottom: 2px solid #00d2fd !important;
}

/* ── Code block ── */
.gradio-container .codemirror-wrapper,
.gradio-container .cm-editor {
    background: #000000 !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── Tip bar ── */
.tip {
    background: #10131a;
    border-radius: 6px;
    padding: 9px 16px;
    margin-top: 1.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem;
    color: #2e3340;
    letter-spacing: .05em;
}
.tip b { color: #45484f; font-weight: 500; }
"""

# ── HTML blocks ───────────────────────────────────────────────────────────

MASTHEAD_HTML = """
<div id="masthead">
  <span class="eyebrow">Computer Vision  ·  Multi-Object Tracking  ·  Applied AI</span>
  <h1>Sports <em>Observer</em></h1>
  <div class="badge-row">
    <span class="bdg bdg-p">YOLOv8n</span>
    <span class="bdg bdg-s">ByteTrack</span>
    <span class="bdg bdg-t">Trajectory Trails</span>
    <span class="bdg bdg-n">Speed Estimation</span>
    <span class="bdg bdg-n">Heatmap</span>
    <span class="bdg bdg-n">HF Spaces</span>
  </div>
</div>
"""

TELEM_HTML = """
<div class="telem">
  <div class="tcard"><div class="tv ca" id="t-ids">—</div><div class="tk">Unique IDs</div></div>
  <div class="tcard"><div class="tv cs" id="t-fr">—</div><div class="tk">Frames</div></div>
  <div class="tcard"><div class="tv ct" id="t-fps">—</div><div class="tk">Source FPS</div></div>
</div>
"""

TIP_HTML = """
<div class="tip">
  <b>TIP</b> &nbsp;·&nbsp; 15–60 s clips give best results on CPU &nbsp;·&nbsp;
  Lower confidence → more detections &nbsp;·&nbsp;
  Works with football, cricket, basketball, athletics footage
</div>
"""


# ══════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
        title="Sports Observer",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:

        gr.HTML(MASTHEAD_HTML)
        gr.HTML('<div class="workspace">')

        # ── LEFT: Control Panel ───────────────────────────────────────────
        gr.HTML('<div class="ctrl">')

        gr.HTML('<div class="sec">Input Stream</div>')
        video_in = gr.Video(label="Upload video", height=210, elem_classes="video-well")

        gr.HTML('<div class="sec">Detection Parameters</div>')
        conf = gr.Slider(0.10, 0.90, value=0.30, step=0.05, label="Confidence threshold")
        iou  = gr.Slider(0.10, 0.90, value=0.50, step=0.05, label="IoU threshold  (NMS)")

        gr.HTML('<div class="sec">Visualisation</div>')
        show_traj  = gr.Checkbox(value=True, label="Trajectory trails")
        show_speed = gr.Checkbox(value=True, label="Speed estimates  (km/h)")
        traj_len   = gr.Slider(10, 120, value=60, step=10, label="Trail length  (frames)")

        run_btn = gr.Button("▶  Run Tracker", elem_id="run-btn", variant="primary")

        gr.HTML('</div>')   # close .ctrl

        # ── RIGHT: Output Panel ───────────────────────────────────────────
        gr.HTML('<div class="out-panel">')

        gr.HTML(TELEM_HTML)

        with gr.Tabs():
            with gr.TabItem("Stream Output"):
                video_out   = gr.Video(label="", height=340, elem_classes="video-well")
            with gr.TabItem("Movement Heatmap"):
                heatmap_out = gr.Image(label="", height=340)
            with gr.TabItem("Telemetry JSON"):
                stats_out   = gr.Textbox(label="Telemetry JSON", lines=16, max_lines=20)

        status_out = gr.HTML("")

        gr.HTML('</div>')   # close .out-panel
        gr.HTML('</div>')   # close .workspace

        gr.HTML(TIP_HTML)

        # ── Wire ─────────────────────────────────────────────────────────
        run_btn.click(
            fn=process_video,
            inputs=[video_in, conf, iou, show_traj, show_speed, traj_len],
            outputs=[video_out, heatmap_out, stats_out, status_out],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
