#!/bin/bash
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

# Launch script for IG info gain service
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset http_proxy https_proxy
export HF_ENDPOINT="https://hf-mirror.com"
set -e

# Default values
HOST="${IG_HOST:-0.0.0.0}"
PORT="${IG_PORT:-0310}"
MODEL_PATH="${IG_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DEVICE="${IG_DEVICE:-cuda:0}"
NUM_GENERATIONS="${IG_NUM_GENERATIONS:-10}"
TEMPERATURE="${IG_TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${IG_MAX_NEW_TOKENS:-128}"
MAX_CONTEXT_WORDS="${IG_MAX_CONTEXT_WORDS:-4096}"
SUB_BATCH_SIZE="${IG_SUB_BATCH_SIZE:-64}"
COMPUTATION_CHUNK_SIZE="${IG_COMPUTATION_CHUNK_SIZE:-256}"
# Recommended: MAX_CONCURRENT_REQUESTS = NUM_GPUS * 5-10
# For 4 GPUs: 20-40 is reasonable. Higher values may cause OOM or slow response.
MAX_CONCURRENT_REQUESTS="${IG_MAX_CONCURRENT_REQUESTS:-512}"
NUM_GPUS="${IG_NUM_GPUS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max-context-words)
            MAX_CONTEXT_WORDS="$2"
            shift 2
            ;;
        --sub-batch-size)
            SUB_BATCH_SIZE="$2"
            shift 2
            ;;
        --computation-chunk-size)
            COMPUTATION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --max-concurrent-requests)
            MAX_CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST                  Host to bind to (default: $HOST)"
            echo "  --port PORT                   Port to bind to (default: $PORT)"
            echo "  --model-path PATH             Path to generator model (default: $MODEL_PATH)"
            echo "  --device DEVICE              Device (e.g., cuda:0) (default: $DEVICE)"
            echo "  --num-generations N           Number of generations (default: $NUM_GENERATIONS)"
            echo "  --temperature T              Generation temperature (default: $TEMPERATURE)"
            echo "  --max-new-tokens N            Max new tokens (default: $MAX_NEW_TOKENS)"
            echo "  --max-context-words N         Max context words (default: $MAX_CONTEXT_WORDS)"
            echo "  --sub-batch-size N            Sub batch size for generation (default: $SUB_BATCH_SIZE)"
            echo "  --computation-chunk-size N    Computation chunk size (default: $COMPUTATION_CHUNK_SIZE)"
            echo "  --max-concurrent-requests N    Max concurrent requests (default: $MAX_CONCURRENT_REQUESTS)"
            echo "  --num-gpus N                  Number of GPUs to use (default: $NUM_GPUS, multi-GPU mode if > 1)"
            echo ""
            echo "Environment variables:"
            echo "  IG_HOST                      Host to bind to"
            echo "  IG_PORT                      Port to bind to"
            echo "  IG_MODEL_PATH                Path to generator model"
            echo "  IG_DEVICE                    Device (e.g., cuda:0)"
            echo "  IG_NUM_GENERATIONS           Number of generations"
            echo "  IG_TEMPERATURE               Generation temperature"
            echo "  IG_MAX_NEW_TOKENS            Max new tokens"
            echo "  IG_MAX_CONTEXT_WORDS         Max context words"
            echo "  IG_SUB_BATCH_SIZE            Sub batch size for generation"
            echo "  IG_COMPUTATION_CHUNK_SIZE    Computation chunk size"
            echo "  IG_MAX_CONCURRENT_REQUESTS   Max concurrent requests"
            echo "  IG_NUM_GPUS                  Number of GPUs to use"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo "Starting IG Info Gain Service..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL_PATH"
echo "  Device: $DEVICE"
echo "  Num generations: $NUM_GENERATIONS"
echo ""

# Export environment variables
export IG_HOST="$HOST"
export IG_PORT="$PORT"
export IG_MODEL_PATH="$MODEL_PATH"
export IG_DEVICE="$DEVICE"
export IG_NUM_GENERATIONS="$NUM_GENERATIONS"
export IG_TEMPERATURE="$TEMPERATURE"
export IG_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export IG_MAX_CONTEXT_WORDS="$MAX_CONTEXT_WORDS"
export IG_SUB_BATCH_SIZE="$SUB_BATCH_SIZE"
export IG_COMPUTATION_CHUNK_SIZE="$COMPUTATION_CHUNK_SIZE"
export IG_MAX_CONCURRENT_REQUESTS="$MAX_CONCURRENT_REQUESTS"
export IG_NUM_GPUS="$NUM_GPUS"

# Run the server
nohup python3 -m IG.service.IG_server \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --num-generations "$NUM_GENERATIONS" \
    --sub-batch-size "$SUB_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-context-words "$MAX_CONTEXT_WORDS" \
    --computation-chunk-size "$COMPUTATION_CHUNK_SIZE" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --num-gpus "$NUM_GPUS" > IG_service_${PORT}.log 2>&1 &
