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

import torch
import numpy as np
import re
import sys
import os
from typing import List, Dict, Any, Optional

# Ensure project root is importable (for IG package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from IG.calculate import (
        gen_answers_batch,
        calculate_uncertainty_soft_batch,
        create_collate_fn,
        process_item_for_ig,
    )
    from IG.models.huggingface_models import HuggingfaceModel
    from IG.uncertainty_measures.semantic_entropy import EntailmentDeberta
except ImportError as e:
    print(f"Warning: Could not import IG modules: {e}")
    print("Please ensure IG is properly installed and accessible")
    gen_answers_batch = None

class IGRewardCalculator:
    """
    IG-based reward calculator for Search-R1 integration.
    Computes ΔIG = IG_with_retrieval - IG_baseline

    Note: Generation model should match the Search-R1 training model for consistent evaluation.
    """

    def __init__(
        self,
        model_path: str = None,  # Will be set to match Search-R1 model
        num_generations: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 128,
        max_context_words: int = 4096,
        computation_chunk_size: int = 8,
        device: str = 'cuda',
        enabled: bool = True
    ):
        self.enabled = enabled

        if not enabled:
            print("IG reward calculation disabled")
            return

        if gen_answers_batch is None:
            raise ImportError("IG modules not available")

        self.model_path = model_path
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_context_words = max_context_words
        self.computation_chunk_size = computation_chunk_size
        self.device = device

        # Initialize models
        try:
            self._init_models()
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            # Fallback to disabled mode
            self.enabled = False
            self.generator = None
            self.entailment_model = None

    def _init_models(self):
        """Initialize generation model and entailment model."""

        # Auto-detect Search-R1 model path if not specified
        if self.model_path is None:
            # Try to get model from environment or use sensible defaults
            self.model_path = os.getenv('ACTOR_MODEL_PATH', None)
            if self.model_path is None:
                # Fallback to common Search-R1 models
                common_models = [
                    'Qwen/Qwen2.5-3B',
                    'Qwen/Qwen2.5-7B',
                    'Qwen/Qwen2.5-1.5B',
                    'meta-llama/Llama-3.2-3B',
                    'meta-llama/Llama-3.1-8B'
                ]
                self.model_path = common_models[0]  # Default to Qwen2.5-3B
                print(f"Auto-detected Search-R1 model: {self.model_path}")
            else:
                print(f"Using model from environment: {self.model_path}")
        else:
            print(f"Using specified IG model: {self.model_path}")

        # Initialize generator model (same as Search-R1 training model)
        try:
            print(f"🔄 Attempting to load {self.model_path}...")
            self.generator = HuggingfaceModel(
                self.model_path,
                stop_sequences='default',
                max_new_tokens=self.max_new_tokens,
                torch_dtype=torch.float16,
                attn_implementation=None,  # Disable flash_attention to avoid issues
                device=self.device,
            )
            self.generator.model.eval()
            print(f"✅ IG generator loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load {self.model_path}: {str(e)[:100]}...")
            print(f"🔄 Trying fallback: Qwen/Qwen2.5-1.5B...")
            try:
                self.model_path = 'Qwen/Qwen2.5-1.5B'
                self.generator = HuggingfaceModel(
                    self.model_path,
                    stop_sequences='default',
                    max_new_tokens=self.max_new_tokens,
                    torch_dtype=torch.float16,
                    attn_implementation=None,
                    device=self.device,
                )
                self.generator.model.eval()
                print(f"✅ IG generator loaded: {self.model_path}")
            except Exception as e2:
                print(f"❌ Failed to load Qwen2.5-1.5B: {str(e2)[:100]}...")
                print("⚠️ Falling back to CPU mode with minimal configuration")
                try:
                    # Final fallback with CPU
                    self.model_path = 'Qwen/Qwen2.5-1.5B'
                    self.generator = HuggingfaceModel(
                        self.model_path,
                        stop_sequences='default',
                        max_new_tokens=min(self.max_new_tokens, 32),  # Reduce for CPU
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        attn_implementation=None,
                        device='cpu',
                    )
                    self.generator.model.eval()
                    print(f"✅ IG generator loaded on CPU: {self.model_path}")
                except Exception as e3:
                    print(f"❌ All model loading attempts failed: {e3}")
                    raise

        # Initialize entailment model
        try:
            print("🔄 Loading entailment model...")
            self.entailment_model = EntailmentDeberta(device=self.device)
            self.entailment_model.model.eval()
            print("✅ IG entailment model loaded: Deberta-v2-xlarge-mnli")
        except Exception as e:
            print(f"❌ Failed to load entailment model: {e}")
            print("🔄 Trying CPU fallback...")
            try:
                self.entailment_model = EntailmentDeberta(device='cpu')
                self.entailment_model.model.eval()
                print("✅ IG entailment model loaded on CPU")
            except Exception as e2:
                print(f"❌ Entailment model loading failed completely: {e2}")
                raise

        # Create collate function
        keys = ['question', 'response_text', 'answers', 'likelihood', 'context_label', 'log_liks_agg', 'context']
        self.ig_collate_fn = create_collate_fn(keys)

        print("✅ IG models initialized successfully")

    def extract_question_and_context(self, sequences_str: str) -> tuple[str, str]:
        """
        Extract question and context from Search-R1 generated sequence.

        Args:
            sequences_str: Full generated sequence with search templates

        Returns:
            Tuple of (question, context)
        """
        # Extract question from the beginning of the sequence
        question_match = re.search(r'Question: (.*?)\n', sequences_str, re.IGNORECASE)
        if question_match:
            question = question_match.group(1).strip()
        else:
            # Fallback: try to extract from prompt structure
            question = sequences_str.split('\n')[0].strip()

        # Extract retrieved context from <information> tags
        info_pattern = r'<information>(.*?)</information>'
        info_matches = re.findall(info_pattern, sequences_str, re.DOTALL)

        if info_matches:
            # Combine all retrieved information as context
            context = '\n'.join([info.strip() for info in info_matches])
        else:
            context = ''

        return question, context

    def extract_ground_truth_answers(self, ground_truth: Dict[str, Any]) -> List[str]:
        """Extract ground truth answers from Search-R1 format."""
        if isinstance(ground_truth, dict):
            return ground_truth.get('target', [])
        elif isinstance(ground_truth, list):
            return ground_truth
        else:
            return [ground_truth]

    def compute_ig_score(self, question: str, context: str, answers: List[str]) -> float:
        """
        Compute IG score for a given question-context pair.

        Args:
            question: The input question
            context: Retrieved context (empty for baseline)
            answers: Ground truth answers

        Returns:
            IG score
        """
        if not self.enabled:
            return 0.0

        # Create example in IG format
        example = {
            'question': question,
            'context': context,
            'answers': answers
        }

        try:
            # Generate answers
            result = gen_answers_batch(
                example,
                self.generator,
                self.temperature,
                self.num_generations,
                sub_batch_size=self.num_generations,
                max_new_tokens=self.max_new_tokens,
                prompt_type='default',
                device=self.device,
                max_context_words=self.max_context_words
            )

            # Convert for IG calculation
            r = process_item_for_ig(result)

            # Calculate IG
            with torch.no_grad():
                ig_input = self.ig_collate_fn([r])
                ig_scores = calculate_uncertainty_soft_batch(
                    ig_input,
                    self.entailment_model,
                    self.computation_chunk_size
                )

            return float(ig_scores[0]) if ig_scores else 0.0

        except Exception as e:
            print(f"Error computing IG score: {e}")
            return 0.0

    def compute_delta_ig_reward(
        self,
        sequences_str: str,
        ground_truth: Dict[str, Any],
        delta_weight: float = 1.0,
        baseline_weight: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute ΔIG reward for Search-R1 integration.

        Args:
            sequences_str: Generated sequence from Search-R1 model
            ground_truth: Ground truth answers
            delta_weight: Weight for ΔIG component
            baseline_weight: Weight for baseline IG component

        Returns:
            Dictionary with IG-based rewards
        """
        if not self.enabled:
            return {'ig_delta': 0.0, 'ig_retrieval': 0.0, 'ig_baseline': 0.0}

        # Extract question and context from generated sequence
        question, context = self.extract_question_and_context(sequences_str)
        answers = self.extract_ground_truth_answers(ground_truth)

        if not question or not answers:
            return {'ig_delta': 0.0, 'ig_retrieval': 0.0, 'ig_baseline': 0.0}

        # Compute IG with retrieved context
        ig_with_retrieval = self.compute_ig_score(question, context, answers)

        # Compute baseline IG (without context)
        ig_baseline = self.compute_ig_score(question, '', answers)

        # Compute ΔIG
        ig_delta = ig_with_retrieval - ig_baseline

        # Combined reward
        ig_reward = delta_weight * ig_delta + baseline_weight * ig_with_retrieval

        return {
            'ig_delta': ig_delta,
            'ig_retrieval': ig_with_retrieval,
            'ig_baseline': ig_baseline,
            'ig_reward': ig_reward
        }


# Global IG calculator instance
_ig_calculator = None

def init_ig_calculator(config: Optional[Dict] = None) -> IGRewardCalculator:
    """
    Initialize global IG calculator.

    Args:
        config: Configuration dictionary with IG settings

    Returns:
        Initialized IGRewardCalculator instance
    """
    global _ig_calculator

    if _ig_calculator is not None:
        return _ig_calculator

    if config is None:
        config = {}

    _ig_calculator = IGRewardCalculator(
        model_path=config.get('model_path', 'meta-llama/Llama-2-7b-chat-hf'),
        num_generations=config.get('num_generations', 5),  # Reduced for efficiency
        temperature=config.get('temperature', 1.0),
        max_new_tokens=config.get('max_new_tokens', 128),
        max_context_words=config.get('max_context_words', 4096),
        computation_chunk_size=config.get('computation_chunk_size', 8),
        device=config.get('device', 'cuda'),
        enabled=config.get('enabled', True)
    )

    return _ig_calculator


def compute_ig_reward(solution_str: str, ground_truth: Dict[str, Any], **kwargs) -> float:
    """
    Compute IG-based reward for Search-R1 integration.

    This function can be called from the existing reward framework.

    Args:
        solution_str: Generated solution from Search-R1
        ground_truth: Ground truth answers
        **kwargs: Additional parameters

    Returns:
        IG reward score
    """
    global _ig_calculator

    if _ig_calculator is None:
        # Auto-initialize with default settings
        init_ig_calculator()

    if _ig_calculator.enabled:
        rewards = _ig_calculator.compute_delta_ig_reward(
            solution_str,
            ground_truth,
            delta_weight=kwargs.get('ig_delta_weight', 1.0),
            baseline_weight=kwargs.get('ig_baseline_weight', 0.0)
        )
        return rewards['ig_reward']
    else:
        return 0.0