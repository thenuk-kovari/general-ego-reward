"""CLI client for a locally served OpenAI-compatible multimodal VLM."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .prompt import SYSTEM_PROMPT, user_prompt

REQUIRED_FIELDS = {
    "reward", "task_progress", "visual_observability", "data_quality",
    "confidence", "success_likelihood", "failure_mode", "evidence",
}


def image_data_url(path: str) -> str:
    image = Path(path)
    if not image.is_file():
        raise FileNotFoundError(f"Frame does not exist: {image}")
    suffix = image.suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported frame type {suffix!r}: {image}")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def parse_score(text: str) -> dict[str, Any]:
    """Parse and minimally validate a strict-JSON model reply."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != REQUIRED_FIELDS:
        raise ValueError(f"Unexpected score schema: {parsed}")
    for key in REQUIRED_FIELDS - {"failure_mode", "evidence"}:
        if not isinstance(parsed[key], int) or not 0 <= parsed[key] <= 100:
            raise ValueError(f"{key} must be an integer from 0 to 100")
    if parsed["failure_mode"] is not None and not isinstance(parsed["failure_mode"], str):
        raise ValueError("failure_mode must be a string or null")
    if not isinstance(parsed["evidence"], str):
        raise ValueError("evidence must be a string")
    return parsed


def score_record(client: OpenAI, model: str, record: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames or not all(isinstance(frame, str) for frame in frames):
        raise ValueError("Each record must contain a non-empty string list named 'frames'")
    task = record.get("task")
    if task is not None and not isinstance(task, str):
        raise ValueError("task must be a string when provided")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt(task, len(frames))}]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url(frame)}} for frame in frames)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    raw = completion.choices[0].message.content
    if raw is None:
        raise RuntimeError("Model returned an empty response")
    result = dict(record)
    result.update(parse_score(raw))
    result["raw_response"] = raw
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL temporal windows")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL scores")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="cyankiwi/InternVL3_5-38B-AWQ-4bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_tokens < 64:
        parser.error("--max-tokens must be at least 64")
    client = OpenAI(base_url=args.endpoint, api_key="local-not-used")
    with args.input.open() as source, args.output.open("w") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL entry must be an object")
                scored = score_record(client, args.model, record, args.max_tokens)
            except Exception as error:
                raise RuntimeError(f"Failed to score {args.input}:{line_number}: {error}") from error
            destination.write(json.dumps(scored) + "\n")


if __name__ == "__main__":
    main()
