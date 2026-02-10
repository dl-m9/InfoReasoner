# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IG HTTP service for computing info gain scores."""

import asyncio
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path to import IG modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IG.calculate import (  # noqa: E402
    calculate_uncertainty_soft_batch,
    create_collate_fn,
    gen_answers_batch,
    process_item_for_ig,
)
from IG.models.huggingface_models import HuggingfaceModel  # noqa: E402
from IG.service.config import IGServiceConfig  # noqa: E402
from IG.uncertainty_measures.semantic_entropy import EntailmentDeberta  # noqa: E402

# Try to import Ray for multi-GPU support
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("[WARNING] Ray not available, multi-GPU mode disabled")

# Import worker for multi-GPU mode
if RAY_AVAILABLE:
    try:
        from IG.service.IG_worker import IGWorker
    except ImportError:
        print("[WARNING] IGWorker not available, multi-GPU mode disabled")
        IGWorker = None
else:
    IGWorker = None


app = FastAPI(title="IG Info Gain Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (for single-GPU mode)
generator: Optional[HuggingfaceModel] = None
entailment_model: Optional[EntailmentDeberta] = None
ig_collate_fn = None
config: Optional[IGServiceConfig] = None

# Thread pool executor for CPU-bound operations
executor: Optional[ThreadPoolExecutor] = None

# Semaphore to limit concurrent GPU operations (prevent OOM)
gpu_semaphore: Optional[asyncio.Semaphore] = None

# Multi-GPU mode (Ray workers)
workers: Optional[List[Any]] = None
worker_index: int = 0  # round-robin
worker_lock: Optional[asyncio.Lock] = None
use_multi_gpu: bool = False


class InfoGainItem(BaseModel):
    """Single item for info gain computation."""

    question: str = Field(..., description="Input question")
    context: str = Field(..., description="Retrieved context (empty for baseline)")
    answers: List[str] = Field(..., description="Ground truth answers")


class InfoGainRequest(BaseModel):
    """Batch request for info gain computation."""

    items: List[InfoGainItem] = Field(..., description="List of items to compute")


class SampleDetail(BaseModel):
    """One sampled response summary."""

    text: str = Field(..., description="Sampled response text")
    mean_loglik: float = Field(..., description="Mean log-likelihood over generated tokens")
    length: int = Field(..., description="Number of log-likelihood entries")


class EntailmentMatrix(BaseModel):
    """Entailment matrices for one example."""

    n_responses: int = Field(..., description="Number of sampled responses")
    n_answers: int = Field(..., description="Number of ground-truth answers")
    resp_entails_ans: List[float] = Field(..., description="Flattened matrix P(resp => ans)")
    ans_entails_resp: List[float] = Field(..., description="Flattened matrix P(ans => resp)")


class ItemDetail(BaseModel):
    """Sample summaries for one item."""

    with_context: List[SampleDetail] = Field(default_factory=list)
    no_context: List[SampleDetail] = Field(default_factory=list)
    entailment_with_context: Optional[EntailmentMatrix] = None
    entailment_no_context: Optional[EntailmentMatrix] = None


class InfoGainResponse(BaseModel):
    """Response containing info gain scores."""

    scores: List[float] = Field(..., description="Info gain scores (Δ)")
    errors: List[Optional[str]] = Field(default_factory=list)
    details: Optional[List[Optional[ItemDetail]]] = None


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global generator, entailment_model, ig_collate_fn, config, executor, gpu_semaphore, workers, use_multi_gpu, worker_lock

    config = IGServiceConfig.from_args()
    worker_lock = asyncio.Lock()

    print("[INFO] Loading IG service with config:")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Num generations: {config.num_generations}")
    print(f"  Max concurrent requests: {config.max_concurrent_requests}")
    print(f"  Num GPUs: {config.num_gpus}")

    if config.num_gpus > 1 and RAY_AVAILABLE and IGWorker is not None:
        print(f"[INFO] Multi-GPU mode enabled: {config.num_gpus} GPUs")
        use_multi_gpu = True

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        print(f"[INFO] Creating {config.num_gpus} Ray workers...")
        workers = []
        for i in range(config.num_gpus):
            worker = IGWorker.remote(
                model_path=config.model_path,
                device=None,
                num_generations=config.num_generations,
                sub_batch_size=config.sub_batch_size,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                max_context_words=config.max_context_words,
                computation_chunk_size=config.computation_chunk_size,
                prompt_type=config.prompt_type,
            )
            workers.append(worker)
            print(f"[INFO] Created worker {i+1}/{config.num_gpus}")

        print("[INFO] Waiting for workers to initialize...")
        ray.get([w.health_check.remote() for w in workers])
        print(f"[INFO] All {len(workers)} workers ready!")

        devices = ray.get([w.get_device.remote() for w in workers])
        print(f"[INFO] Workers using devices: {devices}")
        return

    # Single-GPU mode
    use_multi_gpu = False
    print(f"[INFO] Single-GPU mode (device: {config.device})")

    executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests * 2)
    gpu_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    try:
        print(f"[INFO] Loading generator model: {config.model_path}")
        generator = HuggingfaceModel(
            config.model_path,
            stop_sequences="default",
            max_new_tokens=config.max_new_tokens,
            torch_dtype=torch.float16,
            device=config.device,
        )
        generator.model.eval()
        print("[INFO] Generator model loaded successfully")

        print("[INFO] Loading entailment model...")
        entailment_model = EntailmentDeberta(device=config.device)
        entailment_model.model.eval()
        print("[INFO] Entailment model loaded successfully")

        keys = ["question", "response_text", "answers", "likelihood", "context_label", "log_liks_agg", "context"]
        ig_collate_fn = create_collate_fn(keys)

        print(f"[INFO] IG service ready on {config.device}")
        print(f"[INFO] Async processing enabled: max {config.max_concurrent_requests} concurrent requests")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        traceback.print_exc()
        raise


@app.on_event("shutdown")
async def cleanup():
    """Cleanup resources on shutdown."""
    global executor, workers, use_multi_gpu

    if executor is not None:
        executor.shutdown(wait=True)

    if use_multi_gpu and workers is not None:
        print("[INFO] Shutting down Ray workers...")
        for worker in workers:
            ray.kill(worker)
        print("[INFO] All workers shut down")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global config, workers, use_multi_gpu

    if config is None:
        config = IGServiceConfig.from_args()

    if use_multi_gpu:
        if not workers:
            return {"status": "unhealthy", "message": "No workers available"}
        health_checks = ray.get([w.health_check.remote() for w in workers])
        all_healthy = all(health_checks)
        if all_healthy:
            devices = ray.get([w.get_device.remote() for w in workers])
            return {"status": "healthy", "mode": "multi-GPU", "num_workers": len(workers), "devices": devices}
        return {"status": "unhealthy", "message": "Some workers are unhealthy"}

    if generator is None or entailment_model is None:
        return {"status": "unhealthy", "message": "Models not loaded"}
    return {"status": "healthy", "mode": "single-GPU", "device": config.device if config else "unknown"}


@app.post("/compute_info_gain", response_model=InfoGainResponse)
async def compute_info_gain(request: InfoGainRequest):
    """Compute info gain scores for a batch of items."""
    global workers, use_multi_gpu, worker_index

    items = request.items
    if not items:
        return InfoGainResponse(scores=[], errors=[])

    try:
        if use_multi_gpu and workers:
            if worker_lock is None:
                current_index = 0
            else:
                async with worker_lock:
                    current_index = worker_index
                    worker_index = (worker_index + len(items)) % len(workers)

            ray_refs = []
            for i, item in enumerate(items):
                worker_idx = (current_index + i) % len(workers)
                ref = workers[worker_idx].compute_info_gain.remote(item.question, item.context, item.answers)
                ray_refs.append(ref)

            async def wait_for_result(ref):
                try:
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(None, lambda: ray.get(ref))
                    return score, None
                except Exception as e:
                    return 0.0, f"Error processing item: {str(e)}"

            results = await asyncio.gather(*[wait_for_result(r) for r in ray_refs], return_exceptions=True)
            scores, errors = [], []
            for r in results:
                if isinstance(r, Exception):
                    scores.append(0.0)
                    errors.append(str(r))
                else:
                    score, err = r
                    scores.append(float(score))
                    errors.append(err)
            return InfoGainResponse(scores=scores, errors=errors, details=[None] * len(scores))

        # Single-GPU mode
        if generator is None or entailment_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        if ig_collate_fn is None:
            raise HTTPException(status_code=503, detail="Collate function not initialized")
        if gpu_semaphore is None or executor is None:
            raise HTTPException(status_code=503, detail="Concurrency primitives not initialized")

        async def process_item_async(item: InfoGainItem):
            async with gpu_semaphore:
                loop = asyncio.get_event_loop()
                try:
                    score, detail = await loop.run_in_executor(
                        executor, _compute_single_info_gain_with_details, item.question, item.context, item.answers
                    )
                    return float(score), detail, None
                except Exception as e:
                    traceback.print_exc()
                    return 0.0, None, f"Error processing item: {str(e)}"

        results = await asyncio.gather(*[process_item_async(i) for i in items], return_exceptions=True)

        scores: List[float] = []
        errors: List[Optional[str]] = []
        details: List[Optional[ItemDetail]] = []
        for r in results:
            if isinstance(r, Exception):
                scores.append(0.0)
                errors.append(str(r))
                details.append(None)
            else:
                score, detail, err = r
                scores.append(score)
                errors.append(err)
                details.append(detail)

        return InfoGainResponse(scores=scores, errors=errors, details=details)

    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


def _build_sample_details(result_with_context: Dict[str, Any], result_no_context: Dict[str, Any]) -> ItemDetail:
    def summarize(example: Dict[str, Any]) -> List[SampleDetail]:
        samples: List[SampleDetail] = []
        for text, logliks in example.get("responses", []):
            length = len(logliks)
            mean_ll = float(sum(logliks) / length) if length > 0 else 0.0
            samples.append(SampleDetail(text=str(text).strip(), mean_loglik=mean_ll, length=length))
        return samples

    return ItemDetail(with_context=summarize(result_with_context), no_context=summarize(result_no_context))


def _compute_single_info_gain_with_details(question: str, context: str, answers: List[str]) -> Tuple[float, ItemDetail]:
    global config
    if config is None:
        config = IGServiceConfig.from_args()
    if not isinstance(answers, list):
        answers = [answers]

    generator.model.eval()
    entailment_model.model.eval()

    base_example = {"question": question, "context": context, "answers": answers}

    result_with_context = gen_answers_batch(
        base_example.copy(),
        generator,
        config.temperature,
        config.num_generations,
        config.sub_batch_size,
        config.max_new_tokens,
        config.prompt_type,
        config.device,
        config.max_context_words,
    )
    result_no_context = gen_answers_batch(
        {**base_example, "context": ""},
        generator,
        config.temperature,
        config.num_generations,
        config.sub_batch_size,
        config.max_new_tokens,
        config.prompt_type,
        config.device,
        config.max_context_words,
    )

    with torch.no_grad():
        r_with = process_item_for_ig(result_with_context)
        r_no = process_item_for_ig(result_no_context)
        ig_input = ig_collate_fn([r_with, r_no])
        ig_scores, entailment_details = calculate_uncertainty_soft_batch(
            ig_input,
            entailment_model,
            config.computation_chunk_size,
            return_entailment=True,
        )
        assert len(ig_scores) == 2
        assert len(entailment_details) == 2
        with_ctx, no_ctx = ig_scores
        score = float(with_ctx - no_ctx)

    detail = _build_sample_details(result_with_context, result_no_context)
    with_ctx_ent, no_ctx_ent = entailment_details
    detail.entailment_with_context = EntailmentMatrix(
        n_responses=with_ctx_ent["n_responses"],
        n_answers=with_ctx_ent["n_answers"],
        resp_entails_ans=with_ctx_ent["resp_entails_ans"],
        ans_entails_resp=with_ctx_ent["ans_entails_resp"],
    )
    detail.entailment_no_context = EntailmentMatrix(
        n_responses=no_ctx_ent["n_responses"],
        n_answers=no_ctx_ent["n_answers"],
        resp_entails_ans=no_ctx_ent["resp_entails_ans"],
        ans_entails_resp=no_ctx_ent["ans_entails_resp"],
    )
    return score, detail


def main():
    """Main entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="IG Info Gain Service")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--sub-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-words", type=int, default=None)
    parser.add_argument("--computation-chunk-size", type=int, default=None)
    parser.add_argument("--max-concurrent-requests", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)

    args = parser.parse_args()

    cfg = IGServiceConfig.from_args(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        device=args.device,
        num_generations=args.num_generations,
        sub_batch_size=args.sub_batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_context_words=args.max_context_words,
        computation_chunk_size=args.computation_chunk_size,
        num_gpus=args.num_gpus,
        max_concurrent_requests=args.max_concurrent_requests,
    )

    # Set env vars so startup hook can use them
    if cfg.model_path:
        os.environ["IG_MODEL_PATH"] = cfg.model_path
    if cfg.device:
        os.environ["IG_DEVICE"] = cfg.device
    if cfg.num_generations:
        os.environ["IG_NUM_GENERATIONS"] = str(cfg.num_generations)
    if cfg.sub_batch_size:
        os.environ["IG_SUB_BATCH_SIZE"] = str(cfg.sub_batch_size)
    if cfg.temperature:
        os.environ["IG_TEMPERATURE"] = str(cfg.temperature)
    if cfg.max_new_tokens:
        os.environ["IG_MAX_NEW_TOKENS"] = str(cfg.max_new_tokens)
    if cfg.max_context_words:
        os.environ["IG_MAX_CONTEXT_WORDS"] = str(cfg.max_context_words)
    if cfg.computation_chunk_size:
        os.environ["IG_COMPUTATION_CHUNK_SIZE"] = str(cfg.computation_chunk_size)
    if args.max_concurrent_requests is not None:
        os.environ["IG_MAX_CONCURRENT_REQUESTS"] = str(args.max_concurrent_requests)
    if args.num_gpus is not None:
        os.environ["IG_NUM_GPUS"] = str(args.num_gpus)
    if cfg.host:
        os.environ["IG_HOST"] = cfg.host
    if cfg.port:
        os.environ["IG_PORT"] = str(cfg.port)

    cfg = IGServiceConfig.from_args(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        device=args.device,
        num_generations=args.num_generations,
        sub_batch_size=args.sub_batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_context_words=args.max_context_words,
        computation_chunk_size=args.computation_chunk_size,
        num_gpus=args.num_gpus,
        max_concurrent_requests=args.max_concurrent_requests,
    )

    print(f"[INFO] Starting IG service on {cfg.host}:{cfg.port}")
    print(f"[INFO] Async processing: max {cfg.max_concurrent_requests} concurrent requests")
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1)


if __name__ == "__main__":
    main()

