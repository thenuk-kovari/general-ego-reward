# Ego Progress

An inference-only, zero-shot VLM reward-model baseline for filtering egocentric demonstrations.  It sends a short, ordered frame window and optional task description to InternVL3.5-38B, asks for a structured assessment, and writes one scored record per input.

This is deliberately **not trained**.  The initial question is whether a strong VLM can produce useful, calibrated-enough ranks for data triage before spending time collecting preference labels.

## Model and hardware decision

Use [`cyankiwi/InternVL3_5-38B-AWQ-4bit`](https://huggingface.co/cyankiwi/InternVL3_5-38B-AWQ-4bit) for the first experiment.  InternVL3.5-38B has 38.4B parameters (5.5B vision encoder + 32.8B language model); BF16 weights alone require about 71.5 GiB, before activations and KV cache.  A 40 GB A100 therefore cannot run the BF16 model on one GPU.  Four-bit weights are about 18 GiB before quantization metadata and runtime overhead, making this practical on a single A100 with conservative image and context limits.

The model family’s own documentation says the unquantized 38B deployment needs two A100s.  The AWQ checkpoint is an external conversion, so evaluate it against a small hand-labelled holdout before trusting it for filtering.

## Serve the model on the GPU host

Create an isolated environment on the GPU host and install a CUDA-compatible vLLM release.  Then start an OpenAI-compatible local server:

```bash
pip install vllm
vllm serve cyankiwi/InternVL3_5-38B-AWQ-4bit \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

Keep it bound to loopback and access it through SSH forwarding rather than exposing the server publicly:

```bash
ssh -L 8000:127.0.0.1:8000 USER@GPU_HOST
```

Before the full run, verify `nvidia-smi` reports one complete 40 GB GPU (not a 5 GB MIG slice), and run one request.  If the model fails during startup, reduce `--gpu-memory-utilization` to `0.85`; do not silently fall back to CPU or offload weights.

## Input format

Use JSONL. Each line represents an ordered temporal window:

```json
{"id":"episode-0001","task":"Put the mug on the table","frames":["/data/e0001/000120.jpg","/data/e0001/000150.jpg","/data/e0001/000180.jpg"]}
```

`task` is optional.  Frames must be local image paths on the machine running the client.

## Score a dataset

```bash
python -m ego_progress.score \
  --input windows.jsonl --output scores.jsonl \
  --endpoint http://127.0.0.1:8000/v1 \
  --model cyankiwi/InternVL3_5-38B-AWQ-4bit
```

The output preserves the input fields and adds `reward` (0–100), component scores, a confidence score, evidence, and the raw model answer.  Sort or threshold on `reward`, but use `confidence` and `failure_mode` to route ambiguous clips to review.

## Video reward-progress UI

The web UI samples frames chronologically from an uploaded video, scores each growing prefix, and displays a scalar reward progress bar. It is an intentionally expensive offline baseline: later values can use more temporal evidence than earlier values.

```bash
python -m ego_progress.serve \
  --endpoint http://127.0.0.1:8000/v1 \
  --model cyankiwi/InternVL3_5-38B-AWQ-4bit \
  --host "$(tailscale ip -4)" --port 7860
```

Binding to the Tailscale IPv4 address, rather than `0.0.0.0`, keeps the UI off the public interface. Open `http://TAILSCALE_IP:7860` from another device on the same tailnet. The VLM server itself should remain on `127.0.0.1:8000`.

## Evaluation before filtering

Label 100–300 windows by hand with success/failure or a 1–5 quality score. Compare the scorer's **ranking** (Spearman correlation; AUROC for binary success), inspect the highest-confidence mistakes, and only then choose a filtering threshold.  For offline training data, relative rank is generally more useful than the absolute number.

## Sources

* [InternVL3.5 model card](https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF/blob/main/README.md)
* [InternVL3.5 paper](https://arxiv.org/abs/2508.18265)
* [AWQ-4bit checkpoint and conversion metadata](https://huggingface.co/cyankiwi/InternVL3_5-38B-AWQ-4bit)
* [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/)
