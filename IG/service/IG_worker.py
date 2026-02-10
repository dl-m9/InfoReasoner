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

"""Ray worker for IG info gain computation on a single GPU."""

import gc
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import ray
import torch

# Add parent directory to path to import IG modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IG.calculate import (  # noqa: E402
    calculate_uncertainty_soft_batch,
    create_collate_fn,
    gen_answers_batch,
    process_item_for_ig,
)
from IG.models.vllm_model import VLLMModel  # noqa: E402
from IG.uncertainty_measures.semantic_entropy import EntailmentDeberta  # noqa: E402


@ray.remote(num_gpus=1)
class IGWorker:
    """Ray worker for IG computation on a single GPU."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        num_generations: int = 10,
        sub_batch_size: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 128,
        max_context_words: int = 4096,
        computation_chunk_size: int = 8,
        prompt_type: str = "default",
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"  # Ray sets CUDA_VISIBLE_DEVICES
            else:
                device = "cpu"

        self.device = device
        self.model_path = model_path
        self.num_generations = num_generations
        self.sub_batch_size = sub_batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_context_words = max_context_words
        self.computation_chunk_size = computation_chunk_size
        self.prompt_type = prompt_type

        print(f"[WORKER] Loading models on {device}...")
        try:
            # Use VLLMModel for faster generation
            self.generator = VLLMModel(
                model_path,
                stop_sequences="default",
                max_new_tokens=max_new_tokens,
                device=device,
                gpu_memory_utilization=0.2,  # Leave room for DeBERTa
            )
            self.entailment_model = EntailmentDeberta(device=device)
            self.entailment_model.model.eval()

            keys = ["question", "response_text", "answers", "likelihood", "context_label", "log_liks_agg", "context"]
            self.ig_collate_fn = create_collate_fn(keys)
            print(f"[WORKER] Models loaded successfully on {device}")
        except Exception as e:
            print(f"[WORKER] Failed to load models on {device}: {e}")
            traceback.print_exc()
            raise

    def compute_info_gain(self, question: str, context: str, answers: List[str]) -> float:
        if not isinstance(answers, list):
            answers = [answers]

        self.entailment_model.model.eval()

        example_with_context: Dict[str, Any] = {"question": question, "context": context, "answers": answers}
        result_with_context = gen_answers_batch(
            example_with_context,
            self.generator,
            self.temperature,
            self.num_generations,
            self.sub_batch_size,
            self.max_new_tokens,
            self.prompt_type,
            self.device,
            self.max_context_words,
        )

        result_no_context = gen_answers_batch(
            {**example_with_context, "context": ""},
            self.generator,
            self.temperature,
            self.num_generations,
            self.sub_batch_size,
            self.max_new_tokens,
            self.prompt_type,
            self.device,
            self.max_context_words,
        )

        with torch.no_grad():
            r_with = process_item_for_ig(result_with_context)
            r_no = process_item_for_ig(result_no_context)
            ig_input = self.ig_collate_fn([r_with, r_no])
            with_ctx, no_ctx = calculate_uncertainty_soft_batch(
                ig_input, self.entailment_model, self.computation_chunk_size
            )
            score = float(with_ctx - no_ctx)

        del result_with_context, result_no_context, r_with, r_no, ig_input
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    def compute_info_gain_batch(self, items: List[Dict[str, Any]]) -> List[float]:
        scores: List[float] = []
        for item in items:
            try:
                scores.append(
                    self.compute_info_gain(
                        question=item["question"],
                        context=item.get("context", ""),
                        answers=item["answers"],
                    )
                )
            except Exception as e:
                print(f"[WORKER] Error processing item on {self.device}: {e}")
                traceback.print_exc()
                scores.append(0.0)
        return scores

    def get_device(self) -> str:
        return self.device

    def health_check(self) -> bool:
        return self.generator is not None and self.entailment_model is not None and self.ig_collate_fn is not None

