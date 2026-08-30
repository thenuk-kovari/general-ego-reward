"""Tailscale-bound Gradio interface for zero-shot video reward progress."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
from openai import OpenAI

from .score import score_record


def sample_frames(video_path: str, count: int, directory: Path) -> list[tuple[float, str]]:
    """Extract evenly spaced JPEG frames, returning (timestamp, path) pairs."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not decode the uploaded video")
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if frame_total < 1:
        raise ValueError("The video contains no readable frames")
    indices = sorted({round(i * (frame_total - 1) / max(count - 1, 1)) for i in range(count)})
    sampled: list[tuple[float, str]] = []
    for sample_number, index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        output = directory / f"frame-{sample_number:02d}.jpg"
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Could not write sampled frame {sample_number}")
        sampled.append((index / fps, str(output)))
    capture.release()
    if not sampled:
        raise ValueError("Could not extract any frames from the video")
    return sampled


def progress_html(points: list[dict[str, Any]]) -> str:
    bars = []
    for point in points:
        reward = point["reward"]
        color = "#ef4444" if reward < 35 else "#f59e0b" if reward < 65 else "#22c55e"
        bars.append(
            '<div style="display:grid;grid-template-columns:90px 1fr 48px;gap:8px;align-items:center;margin:8px 0;">'
            f'<span>{point["time_s"]:.1f}s</span>'
            f'<div style="height:18px;background:#e5e7eb;border-radius:5px;overflow:hidden"><div style="height:100%;width:{reward}%;background:{color}"></div></div>'
            f'<b>{reward}</b></div>'
        )
    return "<div><h3>Zero-shot reward progress</h3>" + "".join(bars) + "</div>"


def build_app(endpoint: str, model: str, max_tokens: int) -> gr.Blocks:
    client = OpenAI(base_url=endpoint, api_key="local-not-used")

    def analyze(video: str | None, task: str, samples: int, progress: gr.Progress = gr.Progress()):
        if not video:
            raise gr.Error("Upload a video first.")
        try:
            with tempfile.TemporaryDirectory(prefix="ego-progress-") as temp:
                frames = sample_frames(video, int(samples), Path(temp))
                points = []
                frame_paths: list[str] = []
                for index, (time_s, path) in enumerate(frames, start=1):
                    frame_paths.append(path)
                    progress((index - 1) / len(frames), desc=f"Scoring {index}/{len(frames)}")
                    result = score_record(client, model, {"id": Path(video).stem, "task": task or None, "frames": frame_paths}, max_tokens)
                    points.append({"time_s": round(time_s, 2), "reward": result["reward"], "confidence": result["confidence"], "success_likelihood": result["success_likelihood"], "evidence": result["evidence"], "failure_mode": result["failure_mode"]})
            progress(1.0, desc="Finished")
            return progress_html(points), json.dumps(points, indent=2)
        except Exception as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="Ego Reward Progress") as app:
        gr.Markdown("# Ego reward progress\nUpload a video and describe the intended task. Scores are zero-shot VLM estimates, not calibrated rewards.")
        with gr.Row():
            video = gr.Video(label="Egocentric video", sources=["upload"], type="filepath")
            with gr.Column():
                task = gr.Textbox(label="Task description", placeholder="e.g. Put the mug on the table")
                samples = gr.Slider(2, 8, value=5, step=1, label="Temporal samples")
                submit = gr.Button("Score progress", variant="primary")
        chart = gr.HTML(label="Reward progress")
        details = gr.Code(label="Per-sample details", language="json")
        submit.click(analyze, inputs=[video, task, samples], outputs=[chart, details])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="cyankiwi/InternVL3_5-38B-AWQ-4bit")
    parser.add_argument("--host", required=True, help="Tailscale IPv4 address to bind")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    build_app(args.endpoint, args.model, args.max_tokens).launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
