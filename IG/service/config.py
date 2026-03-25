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

"""Configuration management for IG service."""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class IGServiceConfig:
    """Configuration for IG service."""
    
    # Model configuration
    model_path: str = os.getenv('IG_MODEL_PATH', 'Qwen/Qwen2.5-3B')
    device: str = os.getenv('IG_DEVICE', 'cuda:0')
    
    # Generation parameters
    num_generations: int = int(os.getenv('IG_NUM_GENERATIONS', '10'))
    sub_batch_size: int = int(os.getenv('IG_SUB_BATCH_SIZE', '10'))
    temperature: float = float(os.getenv('IG_TEMPERATURE', '1.0'))
    max_new_tokens: int = int(os.getenv('IG_MAX_NEW_TOKENS', '128'))
    max_context_words: int = int(os.getenv('IG_MAX_CONTEXT_WORDS', '4096'))
    prompt_type: str = os.getenv('IG_PROMPT_TYPE', 'default')
    
    # IG computation parameters
    computation_chunk_size: int = int(os.getenv('IG_COMPUTATION_CHUNK_SIZE', '8'))
    
    # Server configuration
    host: str = os.getenv('IG_HOST', '0.0.0.0')
    port: int = int(os.getenv('IG_PORT', '8000'))
    max_concurrent_requests: int = int(os.getenv('IG_MAX_CONCURRENT_REQUESTS', '4'))
    num_gpus: int = int(os.getenv('IG_NUM_GPUS', '1'))  # Number of GPUs to use (multi-GPU mode if > 1)
    
    @classmethod
    def from_args(cls, **kwargs):
        """Create config from command line arguments."""
        config = cls()
        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config
