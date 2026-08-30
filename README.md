# Ego Progress

Zero-shot reward curves for egocentric robot video using the paper-faithful
[TOPReward](https://arxiv.org/abs/2602.19313) construction. No reward-model
training or generated numeric answer is involved: Qwen3-VL scores the final
single `" True"` token for each growing chronological video prefix, then the
episode is min-max normalized for visualization.

## Local RTX 5080 setup

Qwen3-VL-8B fits on the 16 GB RTX 5080 when loaded as 4-bit NF4. The full
checkpoint occupies about 17 GB on disk; quantized runtime weights plus vision
tokens fit in GPU memory.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[topreward,dev]'
.venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b
```

Score a directory of uniformly sampled JPEG frames:

```bash
.venv/bin/python -m ego_progress.topreward \
  --frames-dir data/frames_00001 \
  --task 'fold and store clothes' \
  --output-dir results/fold_store_00001 \
  --model models/qwen3-vl-8b \
  --load-in-4bit
```

This writes `rewards.json` with raw log-probabilities and
`reward_progress.png` with episode-normalized rewards.

## Extract the GenRobot head camera

Sample `00001` stores H.264 access units on
`/robot0/sensor/camera0/compressed`. Remux the native 30 FPS stream without
re-encoding:

```bash
.venv/bin/python -m ego_progress.extract_mcap_video \
  data/genrobot_00001/00001.mcap \
  results/fold_store_00001/head_camera.mp4
```

## Tailscale web UI

The Gradio UI accepts a video and task, samples chronological frames, scores
every growing prefix, and shows video and reward side by side. During playback,
a live cursor and interpolated reward follow the video clock. Bind only to the
machine.s Tailscale address:

```bash
.venv/bin/python -m ego_progress.topreward_server \
  --model models/qwen3-vl-8b \
  --load-in-4bit \
  --initial-results results/fold_store_00001/rewards.json \
  --initial-video results/fold_store_00001/head_camera.mp4 \
  --host "$(tailscale ip -4)" \
  --port 7860
```

The optional initial files make the synchronized GenRobot sample visible as
soon as the page opens. Uploading a new video runs fresh inference locally.

## Exact TOPReward prompt

The Qwen chat template contains the video followed by:

```text
The above video shows a robot manipulation trajectory that completes the following task: {TASK} Decide whether the above statement is True or not. The answer is: True
```

Only the last token is scored (`" True"`, token ID 3007 for this tokenizer).
The reward is its causal log-probability; per-episode min-max normalization is
used only to make the progress curve readable.

The older structured-judge prototype remains in `ego_progress/score.py` and
`ego_progress/serve.py` for comparison, but TOPReward is the default baseline.

## Evaluation before filtering

Do not interpret a single zero-shot curve as calibrated success probability.
Label a small holdout and evaluate ranking (Spearman correlation, and AUROC for
binary success), inspect high-confidence errors, and only then choose a data
filtering threshold.

## Sources

- [TOPReward paper](https://arxiv.org/abs/2602.19313)
- [Official TOPReward implementation](https://github.com/TOPReward/TOPReward)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
