import logging
from typing import List, Optional, Union, Dict
import torch
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from .base_model import BaseModel, STOP_SEQUENCES

class VLLMModel(BaseModel):
    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, device='auto', **kwargs):
        if LLM is None:
            raise ImportError("vLLM is not installed")
            
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be provided")
        self.max_new_tokens = max_new_tokens
        
        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        self.stop_sequences = stop_sequences if stop_sequences else []
        
        self.model_name = model_name
        
        # Initialize vLLM
        # trust_remote_code is often needed for newer models like Qwen
        # tensor_parallel_size=1 since we are in a single-GPU worker
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto",
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.token_limit = 32768 

    @property
    def model(self):
        # Compatibility with existing code that calls model.model.eval()
        class DummyModel:
            def eval(self):
                pass
        return DummyModel()

    def batch_predict(self, input_data: str, num_generations: int, temperature: float, return_full=False, device='cuda', max_new_tokens=None):
        """
        Generate multiple completions for a single input string.
        """
        prompts = [input_data]
        
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        
        # vLLM sampling parameters
        # We request logprobs=1 to get at least the most likely token logprob.
        # Note: Ideally we want the logprob of the *sampled* token. 
        # vLLM returns a dict of top-k logprobs. If sampled token is not in top-k, we might miss it.
        # But usually k=1 is used in approximate settings or we assume high prob tokens are sampled.
        sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self.stop_sequences,
            logprobs=1 
        )
        
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        output = outputs[0]
        
        sliced_answers = []
        batch_log_likelihoods = []
        
        for completion in output.outputs:
            text = completion.text
            # Remove stop words if vLLM didn't do it perfectly (it usually does stop generation but includes the stop string?)
            # vLLM 'stop' parameter stops generation but does NOT include the stop string in default behavior?
            # Actually vLLM usually stops *before* the stop string or *at* it.
            # Let's strip just in case.
            sliced_answers.append(text.strip())
            
            # Extract logprobs
            if completion.logprobs:
                log_liks = []
                token_ids = completion.token_ids
                for i, token_id in enumerate(token_ids):
                    # completion.logprobs is a list of dicts {token_id: Logprob}
                    # We try to find our token_id
                    token_logprobs_dict = completion.logprobs[i]
                    if token_id in token_logprobs_dict:
                        log_liks.append(token_logprobs_dict[token_id].logprob)
                    elif len(token_logprobs_dict) > 0:
                        # Fallback: take the first (most likely) one
                        log_liks.append(list(token_logprobs_dict.values())[0].logprob)
                    else:
                        log_liks.append(0.0)
                batch_log_likelihoods.append(log_liks)
            else:
                batch_log_likelihoods.append([])
        # Note: vLLM outputs will be cleaned up at worker level after all computations
                
        return sliced_answers, batch_log_likelihoods, None

    def predict(self, input_data, temperature, return_full=False, device='cuda', max_new_tokens=None):
        sliced_answers, batch_log_likelihoods, _ = self.batch_predict(
            input_data, 
            num_generations=1, 
            temperature=temperature, 
            return_full=return_full, 
            device=device, 
            max_new_tokens=max_new_tokens
        )
        return sliced_answers[0], batch_log_likelihoods[0], None

    def get_p_true(self, input_data):
        return 0.0
