"""Gradio web app backed by the paper-faithful TOPReward scorer."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

from .serve import sample_frames
from .topreward import TOPRewardScorer, minmax


def build_app(model_name: str, fps: float) -> gr.Blocks:
    scorer = TOPRewardScorer(model_name)

    def analyze(video: str | None, task: str, samples: int, progress: gr.Progress = gr.Progress()):
        if not video or not task.strip():
            raise gr.Error("Upload a video and provide a task description.")
        with tempfile.TemporaryDirectory(prefix="topreward-") as temp:
            sampled = sample_frames(video, int(samples), Path(temp))
            images = [Image.open(path).convert("RGB") for _, path in sampled]
            raw = []
            metadata = []
            for end in range(2, len(images) + 1):
                progress((end - 2) / max(len(images) - 1, 1), desc=f"Scoring prefix {end - 1}/{len(images) - 1}")
                value, token_id, token_text = scorer.score(images[:end], task.strip(), fps)
                raw.append(value)
                metadata.append({"time_s": sampled[end - 1][0], "log_p_true": value, "token_id": token_id, "token_text": token_text})
        rewards = minmax(raw)
        for item, reward in zip(metadata, rewards, strict=True):
            item["reward"] = reward
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot([x["time_s"] for x in metadata], rewards, marker="o", linewidth=2.5)
        ax.fill_between([x["time_s"] for x in metadata], 0, rewards, alpha=0.15)
        ax.set(title=f'TOPReward: "{task.strip()}"', xlabel="Video time (seconds)", ylabel="Episode-normalized reward", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, json.dumps(metadata, indent=2)

    with gr.Blocks(title="Ego TOPReward") as app:
        gr.Markdown("# Ego TOPReward\nToken-probability progress from chronological video prefixes; no generated numeric scores.")
        with gr.Row():
            video = gr.Video(label="Egocentric video", sources=["upload"], type="filepath")
            with gr.Column():
                task = gr.Textbox(label="Task", placeholder="fold and store clothes")
                samples = gr.Slider(3, 16, value=8, step=1, label="Uniform temporal samples")
                submit = gr.Button("Compute TOPReward", variant="primary")
        plot = gr.Plot(label="Reward versus timestep")
        details = gr.Code(label="Raw log P(True) and normalized reward", language="json")
        submit.click(analyze, inputs=[video, task, samples], outputs=[plot, details])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--host", required=True, help="Tailscale IP to bind")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.model, args.fps).launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
