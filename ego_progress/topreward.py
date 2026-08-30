"""Paper-faithful TOPReward scoring for chronological image prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT_PREFIX = "The above video shows a robot manipulation trajectory that completes the following task: "
PROMPT_SUFFIX = " Decide whether the above statement is True or not. The answer is: True"


class TOPRewardScorer:
    def __init__(self, model_name: str, load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
        ).eval()

    def score(self, frames: list[Any], instruction: str, fps: float) -> tuple[float, int, str]:
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        with torch.inference_mode():
            return self._score(frames, instruction, fps, F, process_vision_info)

    def _score(self, frames: list[Any], instruction: str, fps: float, F: Any, process_vision_info: Any) -> tuple[float, int, str]:
        content = [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": PROMPT_PREFIX},
        ]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        eos = self.processor.tokenizer.eos_token
        if eos:
            prompt = prompt.split(eos)[0]
        full_text = f"{prompt}{instruction}{PROMPT_SUFFIX}"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", cap_pixels_per_frame=True,
        ).to("cuda")
        labels = inputs["input_ids"].clone()
        labels[:, :-1] = -100
        outputs = self.model(**inputs)
        target = labels[:, 1:]
        logits = outputs.logits[:, :-1, :]
        token_id = int(target[0, -1])
        log_probability = float(F.log_softmax(logits[0, -1], dim=-1)[token_id])
        token_text = self.processor.tokenizer.decode([token_id])
        return log_probability, token_id, token_text


def minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--source-fps", type=float, default=30.0, help="Original camera FPS used for plot timestamps")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use NF4 weights for 16 GB GPUs")
    args = parser.parse_args()

    paths = sorted(args.frames_dir.glob("*.jpg"))
    if len(paths) < 2:
        parser.error("--frames-dir must contain at least two JPEG frames")
    frames = [Image.open(path).convert("RGB") for path in paths]
    scorer = TOPRewardScorer(args.model, load_in_4bit=args.load_in_4bit)
    points = []
    raw_rewards = []
    for end in range(2, len(frames) + 1):
        raw, token_id, token_text = scorer.score(frames[:end], args.task, args.fps)
        raw_rewards.append(raw)
        points.append({
            "prefix_frames": end,
            "source_frame": int(paths[end - 1].stem),
            "time_s": int(paths[end - 1].stem) / args.source_fps,
            "log_p_true": raw,
            "token_id": token_id,
            "token_text": token_text,
        })
    normalized = minmax(raw_rewards)
    for point, reward in zip(points, normalized, strict=True):
        point["reward"] = reward

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rewards.json").write_text(json.dumps({
        "task": args.task,
        "model": args.model,
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
        "points": points,
    }, indent=2))
    x = np.array([point["source_frame"] for point in points])
    y = np.array(normalized)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", linewidth=2.5, color="#2563eb")
    ax.fill_between(x, 0, y, alpha=0.15, color="#2563eb")
    ax.set(title=f'TOPReward progress: "{args.task}"', xlabel="Camera timestep (frame)", ylabel="Episode-normalized reward")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "reward_progress.png", dpi=160)


if __name__ == "__main__":
    main()
