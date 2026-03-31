"""
app.py
------
Gradio web interface for Hugging Face Spaces deployment.
UI follows the "Digital Observer" design system (DESIGN.md):
  - Dark void palette: background #0b0e14, surface-container #161a21
  - No 1px borders for sectioning — tonal background shifts only
  - Glassmorphism on floating telemetry panels
  - Space Grotesk for headlines, Inter for data/body
  - Accent: primary #a1ffc2, secondary #00d2fd, tertiary #ff7350
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from detector   import PersonDetector
from tracker    import SportsTracker
from visualizer import FrameVisualizer, HeatmapAccumulator


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def process_video(
    video_path: str,
    conf: float,
    iou: float,
    show_traj: bool,
    show_speed: bool,
    traj_len: int,
    progress=gr.Progress(track_tqdm=True),
):
    if video_path is None:
        return None, None, "{}", _status_html("warning", "Upload a video to begin.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "{}", _status_html("error", "Cannot open video file.")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_raw = tempfile.mktemp(suffix="_raw.mp4")
    tmp_out = tempfile.mktemp(suffix="_out.mp4")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(tmp_raw, fourcc, fps, (width, height))

    detector   = PersonDetector("yolov8n.pt", conf, iou, "cpu")
    tracker    = SportsTracker(fps, conf, iou, traj_len)
    visualizer = FrameVisualizer(show_traj, show_speed)
    heatmap    = HeatmapAccumulator((height, width))

    counts_over_time = []
    frame_idx = 0
    progress(0, desc="Initialising pipeline…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked    = tracker.update(detections)
        heatmap.add(tracked)

        annotated = visualizer.annotate(frame, tracked, tracker.state, frame_idx)
        writer.write(annotated)
        counts_over_time.append({"frame": frame_idx, "count": len(tracked)})

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress(frame_idx / max(total, 1),
                     desc=f"Frame {frame_idx}/{total}  ·  Active IDs: {len(tracked)}")

    cap.release()
    writer.release()

    os.system(
        f'ffmpeg -y -i "{tmp_raw}" -vcodec libx264 -crf 23 -preset fast '
        f'-movflags +faststart "{tmp_out}" -loglevel error'
    )
    if not Path(tmp_out).exists() or Path(tmp_out).stat().st_size == 0:
        tmp_out = tmp_raw

    hm_path = tempfile.mktemp(suffix="_heatmap.png")
    cv2.imwrite(hm_path, heatmap.render())

    uid = len(tracker.state.unique_ids)
    stats = json.dumps({
        "total_frames"     : frame_idx,
        "unique_ids"       : uid,
        "all_track_ids"    : tracker.state.unique_ids,
        "counts_over_time" : counts_over_time[-300:],
    }, indent=2)

    status_html = _status_html(
        "ok",
        f"Stream complete  ·  {frame_idx} frames processed  ·  {uid} unique subjects tracked",
    )
    return tmp_out, hm_path, stats, status_html


def _status_html(kind: str, msg: str) -> str:
    colours = {
        "ok":      ("#a1ffc2", "#00391e"),
        "warning": ("#ffd77a", "#4a3000"),
        "error":   ("#ff716c", "#3d0000"),
    }
    bg, fg = colours.get(kind, colours["ok"])
    return (
        f'<div style="background:{bg}18;border-left:3px solid {bg};'
        f'padding:10px 14px;border-radius:0 6px 6px 0;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
        f'color:{bg};letter-spacing:0.03em;margin-top:8px;">'
        f'<span style="opacity:.6;margin-right:8px;">STATUS</span>{msg}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════
#  CSS  —  "The Digital Observer" design system
# ══════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
body { background: #0b0e14 !important; }
.gradio-container {
    background: #0b0e14 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2.25rem !important;        /* spacing.10 outer margin */
    font-family: 'Inter', sans-serif !important;
}

/* ── Masthead ── */
#masthead {
    background: #161a21;               /* surface-container */
    border-radius: 12px;
    padding: 2rem 2.25rem 1.75rem;
    margin-bottom: 1.75rem;            /* spacing.8 block separation */
    position: relative;
    overflow: hidden;
}
#masthead::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, #a1ffc208 0%, #00fc9a04 50%, transparent 100%);
    pointer-events: none;
}
#masthead h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;                   /* Headline-LG */
    font-weight: 700;
    color: #ecedf6;                    /* on-surface — never pure white */
    letter-spacing: -0.025em;
    margin: 0 0 0.35rem;
    line-height: 1.15;
}
#masthead h1 span {
    color: #a1ffc2;                    /* primary */
}
#masthead .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #5a6475;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}
#masthead .badges {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
#masthead .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid;
    letter-spacing: 0.06em;
}
.badge-primary  { color: #a1ffc2; border-color: #a1ffc230; background: #a1ffc210; }
.badge-secondary{ color: #00d2fd; border-color: #00d2fd30; background: #00d2fd10; }
.badge-tertiary { color: #ff7350; border-color: #ff735030; background: #ff735010; }
.badge-neutral  { color: #6b7585; border-color: #6b758530; background: transparent; }

/* ── Layout columns ── */
.workspace {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.75rem;                      /* spacing.8 */
    align-items: start;
}
@media (max-width: 800px) { .workspace { grid-template-columns: 1fr; } }

/* ── Control panel (left) ── */
.ctrl-panel {
    background: #161a21;               /* surface-container */
    border-radius: 10px;
    padding: 0.9rem;                   /* spacing.4 internal */
    display: flex;
    flex-direction: column;
    gap: 0;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4555;
    padding: 0.9rem 0 0.5rem;
    border-top: 1px solid #1c2028;    /* surface-container-high subtle divider */
    margin-top: 0.5rem;
}
.section-label:first-child {
    border-top: none;
    padding-top: 0.2rem;
    margin-top: 0;
}

/* ── Gradio component overrides ── */
.gradio-container label,
.gradio-container .label-wrap span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #8a95a8 !important;
    font-weight: 400 !important;
}
.gradio-container input[type=range] {
    accent-color: #00d2fd;             /* secondary */
}
.gradio-container input[type=checkbox] {
    accent-color: #a1ffc2;             /* primary */
}

/* ── CTA button ── */
#run-btn button {
    background: #a1ffc2 !important;    /* primary */
    color: #00643a !important;         /* on-primary */
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;    /* rounded-md */
    letter-spacing: 0.02em !important;
    height: 44px !important;
    transition: opacity .15s, transform .1s !important;
    margin-top: 0.9rem;
}
#run-btn button:hover  { opacity: .88 !important; }
#run-btn button:active { transform: scale(0.98) !important; }

/* ── Output panel (right) ── */
.out-panel {
    background: #161a21;
    border-radius: 10px;
    padding: 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
}

/* ── Telemetry cards (glassmorphism floating panels) ── */
.telem-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 0.9rem;
}
.telem-card {
    background: rgba(34, 38, 47, 0.60);   /* surface-variant 60% */
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 8px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}
.telem-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #a1ffc2, #00fc9a);
    opacity: 0.10;
}
.telem-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ecedf6;
    line-height: 1;
    margin-bottom: 4px;
}
.telem-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3d4555;
}
.telem-accent { color: #a1ffc2; }      /* primary */
.telem-secondary { color: #00d2fd; }   /* secondary */
.telem-alert { color: #ff7350; }       /* tertiary */

/* ── Tabs ── */
.gradio-container .tab-nav {
    background: #10131a !important;    /* surface-container-low */
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
    padding: 0 4px !important;
}
.gradio-container .tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    color: #3d4555 !important;
    border: none !important;
    padding: 10px 16px !important;
}
.gradio-container .tab-nav button.selected {
    color: #00d2fd !important;          /* secondary */
    border-bottom: 2px solid #00d2fd !important;
    background: transparent !important;
}

/* ── Video player ── */
.gradio-container video {
    border-radius: 6px;
    background: #000000;               /* surface-container-lowest recessed well */
}

/* ── Active stream indicator ── */
.stream-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #a1ffc2;               /* primary */
    margin-right: 6px;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(.75); }
}

/* ── Footer tip ── */
.tip-bar {
    background: #10131a;
    border-radius: 8px;
    padding: 10px 16px;
    margin-top: 1.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3d4555;
    letter-spacing: 0.04em;
}
.tip-bar strong { color: #5a6475; font-weight: 500; }

/* ── Heatmap/image output ── */
.gradio-container .image-container {
    border-radius: 6px;
    overflow: hidden;
}

/* ── Code block ── */
.gradio-container .codemirror-wrapper {
    background: #000000 !important;
    border-radius: 6px !important;
}
"""

# ── Masthead HTML ─────────────────────────────────────────────────────────
MASTHEAD = """
<div id="masthead">
  <p class="sub">Computer Vision · Multi-Object Tracking · Applied AI</p>
  <h1>Sports <span>Observer</span></h1>
  <div class="badges">
    <span class="badge badge-primary">YOLOv8n</span>
    <span class="badge badge-secondary">ByteTrack</span>
    <span class="badge badge-tertiary">Trajectory</span>
    <span class="badge badge-neutral">Speed Estimation</span>
    <span class="badge badge-neutral">Heatmap</span>
  </div>
</div>
"""

TELEMETRY_INIT = """
<div class="telem-row">
  <div class="telem-card">
    <div class="telem-val telem-accent" id="t-ids">—</div>
    <div class="telem-key">Unique IDs</div>
  </div>
  <div class="telem-card">
    <div class="telem-val telem-secondary" id="t-frames">—</div>
    <div class="telem-key">Frames</div>
  </div>
  <div class="telem-card">
    <div class="telem-val telem-alert" id="t-fps">—</div>
    <div class="telem-key">Source FPS</div>
  </div>
</div>
"""

TIP = """
<div class="tip-bar">
  <strong>TIP ·</strong>&nbsp; 15–60 s clips work best on CPU free tier &nbsp;·&nbsp;
  Lower confidence → more detections &nbsp;·&nbsp;
  Works with cricket, football, basketball, athletics
</div>
"""


# ══════════════════════════════════════════════════════════════════════════
#  GRADIO APP
# ══════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
        title="Sports Observer — Multi-Object Tracker",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        gr.HTML(MASTHEAD)
        gr.HTML('<div class="workspace">')   # open grid wrapper

        # ── LEFT: Control Panel ───────────────────────────────────────────
        gr.HTML('<div class="ctrl-panel">')

        gr.HTML('<div class="section-label">Input stream</div>')
        video_in = gr.Video(label="Upload video", height=220)

        gr.HTML('<div class="section-label">Detection parameters</div>')
        conf = gr.Slider(0.10, 0.90, value=0.30, step=0.05,
                         label="Confidence threshold")
        iou  = gr.Slider(0.10, 0.90, value=0.50, step=0.05,
                         label="IoU threshold  (NMS)")

        gr.HTML('<div class="section-label">Visualisation</div>')
        traj     = gr.Checkbox(value=True, label="Trajectory trails")
        speed    = gr.Checkbox(value=True, label="Speed estimates  (km/h)")
        traj_len = gr.Slider(10, 120, value=60, step=10,
                             label="Trail length  (frames)")

        run_btn = gr.Button("Run Tracker", elem_id="run-btn", variant="primary")

        gr.HTML('</div>')  # close ctrl-panel

        # ── RIGHT: Output Panel ───────────────────────────────────────────
        gr.HTML('<div class="out-panel">')

        gr.HTML(TELEMETRY_INIT)

        with gr.Tabs():
            with gr.TabItem("Stream Output"):
                video_out = gr.Video(label="", height=320)
            with gr.TabItem("Movement Heatmap"):
                heatmap_out = gr.Image(label="", height=320)
            with gr.TabItem("Telemetry JSON"):
                stats_out = gr.Code(language="json", label="", lines=14)

        status_out = gr.HTML("")

        gr.HTML('</div>')  # close out-panel
        gr.HTML('</div>')  # close workspace grid

        gr.HTML(TIP)

        # ── Wire ─────────────────────────────────────────────────────────
        run_btn.click(
            fn=process_video,
            inputs=[video_in, conf, iou, traj, speed, traj_len],
            outputs=[video_out, heatmap_out, stats_out, status_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
