# *InfoReasoner*: Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward

<p align="center">
  <a href="https://arxiv.org/abs/2602.00845">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.00845-b31b1b.svg" />
  </a>
  <a href="https://huggingface.co/papers/2602.00845">
    <img alt="Hugging Face Daily Paper" src="https://img.shields.io/badge/HF%20Daily%20Paper-2602.00845-FFCC00?logo=huggingface&logoColor=000" />
  </a>
</p>

## Abstract

Agentic reasoning enables large reasoning models (LRMs) to dynamically acquire external knowledge, but yet optimizing the retrieval process remains challenging due to the lack of dense, principled reward signals. In this paper, we introduce InfoReasoner, a unified framework that incentivizes effective information seeking via a synthetic semantic information gain reward. Theoretically, we redefine information gain as uncertainty reduction over the model's belief states, establishing guarantees, including non-negativity, telescoping additivity, and channel monotonicity. Practically, to enable scalable optimization without manual retrieval annotations, we propose an output-aware intrinsic estimator that computes information gain directly from the model's output distributions using semantic clustering via bidirectional textual entailment. This intrinsic reward guides the policy to maximize epistemic progress, enabling efficient training via Group Relative Policy Optimization (GRPO). Experiments across seven question-answering benchmarks demonstrate that InfoReasoner consistently outperforms strong retrieval-augmented baselines, achieving up to 5.4% average accuracy improvement. Our work provides a theoretically grounded and scalable path toward agentic reasoning with retrieval.



## Project Overview

This repository contains the code for **InfoReasoner**, including:

- An **information-gain (IG) reward service** implemented as a FastAPI server under `IG/`.
- A **training pipeline** built on top of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/volcengine/verl), using GRPO and the synthetic semantic IG reward.
- Shell scripts and configs to reproduce GRPO training with IG on retrieval-augmented QA datasets.

At a high level, the workflow is:

1. Launch a **retriever server** (for document retrieval, as in Search-R1).
2. Launch the **IG service** (computes semantic information gain given question / context / answers).
3. Run **GRPO training** with the IG reward, via `train_grpo.sh` (which wraps `verl/trainer/config/ppo_trainer.yaml` with overrides).

## Environment Setup

We recommend **three conda environments**:

- One for **RL training** (Search-R1 + veRL).
- One for the **retriever service** (Faiss / retrieval stack).
- One for the **IG service**.

### 1. Training environment (Search-R1 / veRL)

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

# Core dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3

# Install this repo (Search-R1 + InfoReasoner)
pip install -e .

# Optional but recommended
pip install flash-attn --no-build-isolation
pip install wandb
```

You can find more details about veRL itself in `VERL_README.md`.

### 2. IG service environment

The IG service runs as a separate HTTP server and can be hosted on a different machine.

```bash
conda create -n ig-service python=3.10
conda activate ig-service

# Core runtime
pip install "torch>=2.1.0"
pip install transformers accelerate sentencepiece

# Service and networking
pip install fastapi uvicorn requests

# Optional: multi-GPU support
pip install ray
```

Make sure the IG service environment has access to the base model checkpoint used for reward computation (e.g., `Qwen/Qwen2.5-3B`).

### 3. Retriever service environment (optional but recommended)

If you run local retrieval (e.g., dense e5 + Faiss), use a dedicated environment:

```bash
conda create -n retriever python=3.10
conda activate retriever

# Recommended for GPU Faiss
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# API server deps
pip install fastapi uvicorn
```

## Retriever Setup (Build + Serve)

This project expects a retriever endpoint like `http://127.0.0.1:8000/retrieve`.

### 1. Prepare corpus and index

You have two common options:

#### Option A: use the provided corpus/index workflow

```bash
save_path=/path/to/retrieval_assets
python scripts/download.py --save_path "$save_path"
cat "$save_path"/part_* > "$save_path"/e5_Flat.index
gzip -d "$save_path"/wiki-18.jsonl.gz
```

Then place assets where your launch script expects them, or pass explicit paths to the server.

#### Option B: build your own index

```bash
bash search_r1/search/build_index.sh
```

Customize retriever model / corpus settings in that script before running.

### 2. Launch retriever server

Quick start with the provided launcher:

```bash
conda activate retriever
bash retrieval_launch.sh
```

Or start manually with explicit paths:

```bash
python search_r1/search/retrieval_server.py \
  --index_path /path/to/e5_Flat.index \
  --corpus_path /path/to/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
```

Default server bind is `0.0.0.0:8000`.

### 3. Verify retriever

Health/behavior can be checked by sending retrieval requests from training or from a simple curl:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?","topk":3}'
```

### 4. Connect retriever to training

In training scripts/configs, ensure:

- `retriever.url` points to your running retriever service (typically `http://127.0.0.1:8000/retrieve`)
- `retriever.topk` matches your desired retrieval depth

For example, `train_grpo.sh` already sets:

```bash
retriever.url="http://127.0.0.1:8000/retrieve"
retriever.topk=3
```

## End-to-End Quickstart (Retriever + IG + GRPO)

The following sequence launches the full stack in three terminals.

### Terminal 1: start retriever service

```bash
conda activate retriever
bash retrieval_launch.sh
```

Expected endpoint: `http://127.0.0.1:8000/retrieve`

### Terminal 2: start IG service

```bash
conda activate ig-service

bash IG_service_launch.sh \
  --port 310 \
  --device cuda:0 \
  --model-path Qwen/Qwen2.5-3B \
  --num-generations 10 \
  --max-concurrent-requests 4 \
  --num-gpus 1
```

Health check:

```bash
curl http://127.0.0.1:310/health
```

### Terminal 3: start GRPO training

```bash
conda activate searchr1

# Optional overrides before launch
export BASE_MODEL=/path/to/base-or-checkpoint
export EXPERIMENT_NAME="$(date +%m%d-%H%M)-nq-train-grpo-ig"

bash train_grpo.sh
```

### Common issues checklist

- Retriever not reachable: verify `retriever.url` in training config points to `:8000/retrieve`.
- IG timeout: increase `IG_TIMEOUT` and/or reduce `IG_BATCH_SIZE`.
- GPU OOM in IG service: reduce `--max-concurrent-requests` or `--num-generations`.
- Throughput bottleneck: use IG multi-GPU mode (`--num-gpus > 1`) and ensure sufficient request concurrency.

## Running the IG Service

The IG service lives under `IG/service/` and is launched via `IG_service_launch.sh`.

### Single-GPU example

```bash
conda activate ig-service

bash IG_service_launch.sh \
  --port 310 \
  --device cuda:0 \
  --model-path Qwen/Qwen2.5-3B \
  --num-generations 10 \
  --max-concurrent-requests 4 \
  --num-gpus 1
```

Key arguments / environment variables (mapped to `IG/service/config.py`):

- `IG_HOST` / `--host` (default `0.0.0.0`)
- `IG_PORT` / `--port` (default `8000`)
- `IG_MODEL_PATH` / `--model-path` (e.g., `Qwen/Qwen2.5-3B`)
- `IG_DEVICE` / `--device` (e.g., `cuda:0`)
- `IG_NUM_GENERATIONS` / `--num-generations`
- `IG_TEMPERATURE` / `--temperature`
- `IG_MAX_NEW_TOKENS` / `--max-new-tokens`
- `IG_MAX_CONTEXT_WORDS` / `--max-context-words`
- `IG_COMPUTATION_CHUNK_SIZE` / `--computation-chunk-size`
- `IG_MAX_CONCURRENT_REQUESTS` / `--max-concurrent-requests`
- `IG_NUM_GPUS` / `--num-gpus`

After startup, you should see logs indicating the generator and entailment models are loaded and the service is ready.

### Multi-GPU mode

The service supports multi-GPU deployment via **Ray**. You can either specify `--num-gpus` on the command line or via `IG_NUM_GPUS`.

```bash
conda activate ig-service

# Use 4 GPUs with Ray workers
bash IG_service_launch.sh \
  --port 310 \
  --num-gpus 4 \
  --model-path Qwen/Qwen2.5-3B
```

Or with environment variables:

```bash
export IG_NUM_GPUS=4
export IG_PORT=310

bash IG_service_launch.sh
```

In multi-GPU mode:

- The main FastAPI process forwards requests to multiple Ray workers.
- Each worker owns one GPU and runs its own generator + entailment model.
- Requests are distributed in a round-robin fashion across workers.

You can verify the deployment:

```bash
curl http://localhost:310/health
```

Multi-GPU responses include fields like:

```json
{
  "status": "healthy",
  "mode": "multi-GPU",
  "num_workers": 4,
  "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
}
```

See `IG/service/README.md` for more details on the service API and configuration.

## Training with IG Reward

Once the retriever and IG service are running, you can launch GRPO training from the project root.

### 1. Configure the base model and logging

Edit `train_grpo.sh` or export the relevant variables:

```bash
export BASE_MODEL=/path/to/base-or-checkpoint
export WAND_PROJECT='Qwen2.5-7B-GRPO'
export EXPERIMENT_NAME="$(date +%m%d-%H%M)-nq_train-grpo-qwen2.5-7b-ig"
```

`train_grpo.sh` also sets:

```bash
export IG_SERVICE_URL="http://0.0.0.0:310"
export IG_BATCH_SIZE='512'
export IG_TIMEOUT='120'
```

Make sure `IG_SERVICE_URL` matches the host/port where you launched the IG service.

### 2. Launch training

```bash
conda activate searchr1

bash train_grpo.sh
```

This script ultimately calls:

- `verl.trainer.main_ppo` with overrides on top of the base Hydra config `verl/trainer/config/ppo_trainer.yaml`.

The actual IG reward computation flows through:

- `verl/utils/reward_score/qa_em.py` (EM + IG integration).
- `verl/utils/reward_score/IG_client.py` (HTTP client wrapper).
- `verl/utils/reward_score/IG_reward.py` (local IG reward calculator, if needed).

### 3. Monitoring runs

`train_grpo.sh` configures Weights & Biases logging via:

- `trainer.project_name`
- `trainer.experiment_name`

You can monitor:

- EM accuracy metrics.
- IG reward statistics (mean / max / min).
- Training losses and KL metrics.

## IG Service API (Quick Reference)

The IG service exposes two main endpoints:

- `GET /health` – basic health and mode information (single-GPU / multi-GPU).
- `POST /compute_info_gain` – batch IG computation.

Example request:

```bash
curl -X POST http://localhost:310/compute_info_gain \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is its capital.",
        "answers": ["Paris"]
      }
    ]
  }'
```

Example response:

```json
{
  "scores": [0.123],
  "errors": [null],
  "details": [...]
}
```

The training code (`qa_em.py`) consumes only the `scores` field; `details` is useful for analysis and ablations.


## Acknowledge

This project builds upon the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework and the [verl](https://github.com/verl-project/verl) reinforcement learning library. We sincerely thank the authors of these projects for their valuable contributions, which have significantly supported and inspired our work.



## Citation

```bibtex
@misc{hu2026optimizingagenticreasoningretrieval,
      title={Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward}, 
      author={Senkang Hu and Yong Dai and Yuzhi Zhao and Yihang Tao and Yu Guo and Zhengru Fang and Sam Tak Wu Kwong and Yuguang Fang},
      year={2026},
      eprint={2602.00845},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.00845}, 
}
```

