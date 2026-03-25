# IG Info Gain Service

This directory contains the **Info Gain (IG) HTTP service** used by InfoReasoner to compute semantic information gain rewards during training.

The service:

- Receives batches of `{question, context, answers}`.
- Samples answers from a generator LLM with and without context.
- Uses a DeBERTa entailment model to estimate semantic uncertainty.
- Returns **ΔIG = IG_with_context − IG_baseline** for each item.

The training code talks to this service over HTTP; when the service is unavailable, the reward code can fall back to local computation.

## 1. Installation

We recommend a dedicated environment for the IG service:

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

Make sure the environment has access to:

- A generator model (e.g., `Qwen/Qwen2.5-3B`).
- A DeBERTa entailment model (see `IG/uncertainty_measures/semantic_entropy.py`).

## 2. Configuration

Runtime configuration is handled by `IG/service/config.py` via the `IGServiceConfig` dataclass.

### 2.1 Environment variables

| Parameter                 | Env var                       | Default                   | Description                                |
|--------------------------|------------------------------|---------------------------|--------------------------------------------|
| Host                     | `IG_HOST`                    | `0.0.0.0`                 | Service bind address                       |
| Port                     | `IG_PORT`                    | `8000`                    | Service port                               |
| Model path               | `IG_MODEL_PATH`              | `Qwen/Qwen2.5-3B`         | Generator model path                       |
| Device                   | `IG_DEVICE`                  | `cuda:0`                  | GPU device for single-GPU mode             |
| Num generations          | `IG_NUM_GENERATIONS`         | `10`                      | Number of samples per item                 |
| Sub-batch size           | `IG_SUB_BATCH_SIZE`          | `10`                      | Generation batch size                      |
| Temperature              | `IG_TEMPERATURE`             | `1.0`                     | Sampling temperature                       |
| Max new tokens           | `IG_MAX_NEW_TOKENS`          | `128`                     | Maximum generated tokens                   |
| Max context words        | `IG_MAX_CONTEXT_WORDS`       | `4096`                    | Max context length (words)                 |
| Prompt type              | `IG_PROMPT_TYPE`             | `default`                 | Prompt template type                       |
| Computation chunk size   | `IG_COMPUTATION_CHUNK_SIZE`  | `8`                       | Entailment batch size                      |
| Max concurrent requests  | `IG_MAX_CONCURRENT_REQUESTS` | `4`                       | Max concurrent requests per process        |
| Number of GPUs           | `IG_NUM_GPUS`                | `1`                       | GPUs to use (`>1` enables multi-GPU mode)  |

These can be overridden by CLI flags to `IG_server.py` (via `IG_service_launch.sh`).

## 3. Running the Service

The recommended entrypoint is the shell script at the repo root:

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

Alternatively, you can configure everything via environment variables:

```bash
export IG_PORT=310
export IG_DEVICE=cuda:0
export IG_MODEL_PATH=Qwen/Qwen2.5-3B
export IG_NUM_GENERATIONS=10

bash IG_service_launch.sh
```

At startup, the server:

1. Loads `IGServiceConfig` from env / CLI.
2. Initializes the generator and entailment models.
3. Exposes the FastAPI app on `IG_HOST:IG_PORT`.

## 4. Single-GPU vs Multi-GPU

The service supports both **single-GPU** and **multi-GPU** deployments.

### 4.1 Single-GPU mode (default)

- Uses local models in the main process.
- Concurrency is handled via:
  - An `asyncio.Semaphore` (`IG_MAX_CONCURRENT_REQUESTS`).
  - A thread pool for CPU/GPU-bound work.
- Suitable for single-GPU setups or moderate throughput.

Example:

```bash
bash IG_service_launch.sh \
  --port 310 \
  --device cuda:0 \
  --num-gpus 1
```

### 4.2 Multi-GPU mode with Ray

When `IG_NUM_GPUS > 1` (or `--num-gpus > 1`) and `ray` is installed:

- The main FastAPI process becomes a **front-end**.
- Multiple `IGWorker` Ray actors are created, each bound to one GPU.
- Requests are split across workers in a round-robin fashion.

Example (4 GPUs):

```bash
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

**Behavior in multi-GPU mode:**

- Each worker owns its generator + entailment model and processes items independently.
- The frontend distributes items evenly across workers.
- Health checks aggregate the status of all workers.

Example health check:

```bash
curl http://localhost:310/health
```

Typical multi-GPU response:

```json
{
  "status": "healthy",
  "mode": "multi-GPU",
  "num_workers": 4,
  "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
}
```

### 4.3 Tuning concurrency

Key knobs:

- `--max-concurrent-requests` / `IG_MAX_CONCURRENT_REQUESTS`: concurrent requests per process.
- `--num-gpus` / `IG_NUM_GPUS`: number of Ray workers (multi-GPU only).

Examples:

- 4 GPUs, 4 concurrent requests per GPU → ~16 effective concurrent requests.
- If you see OOMs, **reduce** `IG_MAX_CONCURRENT_REQUESTS` and/or `IG_NUM_GENERATIONS`.

## 5. HTTP API

### 5.1 Health check

```bash
curl http://localhost:310/health
```

Single-GPU response example:

```json
{
  "status": "healthy",
  "mode": "single-GPU",
  "device": "cuda:0"
}
```

### 5.2 Compute Info Gain

Endpoint:

- `POST /compute_info_gain`

Request body:

```json
{
  "items": [
    {
      "question": "What is the capital of France?",
      "context": "France is a country in Europe. Paris is its capital.",
      "answers": ["Paris"]
    }
  ]
}
```

Response:

```json
{
  "scores": [0.123],
  "errors": [null],
  "details": [...]
}
```

- `scores[i]` is ΔIG for the `i`-th item.
- `errors[i]` is an error message or `null`.
- `details` (single-GPU only) contains:
  - Per-sample summaries (texts, mean log-likelihoods, lengths).
  - Optional entailment matrices for offline analysis.

## 6. Using the Service from Training Code

On the training side, the integration is handled by:

- `verl/utils/reward_score/qa_em.py` – EM + IG reward logic.
- `verl/utils/reward_score/IG_client.py` – thin wrapper that re-exports `IGClient` and helpers.

To point training to the service, set:

```bash
export IG_SERVICE_URL=http://0.0.0.0:310
export IG_BATCH_SIZE=512
export IG_TIMEOUT=120
```

`IG_BATCH_SIZE` and `IG_TIMEOUT` are used by the training code to control:

- How many items are sent per batch.
- How long to wait for a batch before timing out.

## 7. Troubleshooting

### 7.1 Service fails to start

1. Check GPU visibility: `nvidia-smi`.
2. Verify that `IG_MODEL_PATH` exists and can be loaded by `transformers`.
3. Ensure `torch`, `transformers`, `fastapi`, `uvicorn` (and `ray` for multi-GPU) are installed in the IG environment.
4. Look at the console logs for stack traces.

### 7.2 Training cannot reach the service

1. Confirm the service is running: `curl http://HOST:PORT/health`.
2. Verify `IG_SERVICE_URL` matches the host/port where the service is listening.
3. Check firewall / container networking rules.

### 7.3 Performance issues

- Increase `IG_COMPUTATION_CHUNK_SIZE` to speed up entailment at the cost of more memory.
- Use multi-GPU mode (`IG_NUM_GPUS > 1`) for higher throughput.
- Adjust `IG_MAX_CONCURRENT_REQUESTS` to balance concurrency vs. memory.

### 7.4 Multi-GPU issues

- Make sure `ray` is installed and importable.
- Verify the requested number of GPUs is available.
- If performance does not scale, check:
  - Actual request concurrency from the client side.
  - GPU utilization via `nvidia-smi`.
  - Whether `IG_MAX_CONCURRENT_REQUESTS` is too low or too high.

