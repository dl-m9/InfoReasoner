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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import re

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import qa_em
from verl.utils.reward_score.qa_em import get_last_reward_metrics
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Collect detailed reward metrics
        info_gain_scores = []
        output_scores = []  # EM check scores only
        all_scores = []  # Weighted sums

        already_print_data_sources = {}
        
        # Check if this is validation mode and set thread-local flag
        from verl.utils.reward_score import qa_em
        is_validation = data.meta_info.get('validate', False)
        
        # Debug: print meta_info to see what's being passed
        print(f"[DEBUG] data.meta_info keys: {list(data.meta_info.keys())}")
        print(f"[DEBUG] data.meta_info.get('validate'): {data.meta_info.get('validate', 'NOT_FOUND')}")
        print(f"[DEBUG] is_validation: {is_validation}")
        
        if is_validation:
            qa_em._validation_flag.skip_info_gain = True
            print(f"[INFO] Validation mode detected: skipping info gain computation")
        else:
            # Reset flag if not validation
            if hasattr(qa_em._validation_flag, 'skip_info_gain'):
                qa_em._validation_flag.skip_info_gain = False

        # ===== PHASE 1: Collect items for batch info gain computation =====
        batch_items = []  # Items to send to IG service
        batch_indices = []  # Original indices of items that need info gain computation
        item_to_data = []  # Store decoded data for each item
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Store decoded data for later use
            item_to_data.append({
                'sequences_str': sequences_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'compute_score_fn': compute_score_fn,
                'valid_response_length': valid_response_length,
            })

            # Collect items that need info gain computation
            if (compute_score_fn.__name__ == 'compute_score_em' and 
                not is_validation and 
                qa_em.IG_CLIENT_AVAILABLE):
                result = qa_em.extract_question_information(solution_str=sequences_str)
                information_blocks = result['information_blocks']
                question = result['question']
                
                # Only add if there are information blocks
                if len(information_blocks) > 0:
                    context = "\n".join(information_blocks)
                    batch_items.append({
                        'question': question,
                        'context': context,
                        'answers': ground_truth['target'],
                    })
                    batch_indices.append(i)
        # ===== PHASE 2: Batch compute info gain scores =====
        info_gain_scores_dict = {}  # Map from index to info_gain_score
        
        if is_validation:
            print(f"[INFO] Validation mode: skipping batch info gain computation (only using EM scores)")
        elif len(batch_items) > 0:
            # Get IG client
            from verl.utils.reward_score.IG_client import get_ig_client, IGClient
            service_url = os.getenv('IG_SERVICE_URL', 'http://0.0.0.0:0310')
            print(f"[INFO] IG service URL: {service_url}")
            # Configure batch size and timeout
            batch_size = int(os.getenv('IG_BATCH_SIZE', '256'))  # Items per batch
            base_timeout = float(os.getenv('IG_TIMEOUT', '60.0'))  # Base timeout in seconds
            print(f"[INFO] IG batch size: {batch_size}")
            # Calculate timeout: base_timeout per 100 items, with minimum 60s
            # For a batch of 100 items, use 60s; for 200 items, use 120s, etc.
            timeout = max(base_timeout, base_timeout * (batch_size / 128.0))
            
            # Get or create client with increased timeout for batch processing
            ig_client = get_ig_client(service_url=service_url)
            original_timeout = None
            if ig_client is None:
                # Create a new client with custom timeout if global client doesn't exist
                ig_client = IGClient(service_url=service_url, timeout=timeout)
            else:
                # Temporarily increase timeout for the global client
                original_timeout = ig_client.timeout
                ig_client.timeout = timeout
            
            if ig_client is not None and ig_client.health_check():
                try:
                    # Split into smaller batches to avoid timeout
                    # Process in chunks to prevent single request timeout
                    chunk_size = batch_size
                    total_items = len(batch_items)
                    
                    print(f"[INFO] Processing {total_items} items in batches of {chunk_size} (timeout={timeout:.1f}s per batch)")
                    
                    for chunk_start in range(0, total_items, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_items)
                        chunk_items = batch_items[chunk_start:chunk_end]
                        chunk_indices = batch_indices[chunk_start:chunk_end]
                        
                        try:
                            # Batch compute info gain scores for this chunk
                            import time
                            start_time = time.time()

                            chunk_scores = ig_client.compute_info_gain_batch(chunk_items)

                            elapsed_time = time.time() - start_time
                            
                            # Map scores back to original indices
                            for idx, score in zip(chunk_indices, chunk_scores):
                                info_gain_scores_dict[idx] = score
                                
                            print(f"[INFO] Processed chunk {chunk_start//chunk_size + 1}/{(total_items-1)//chunk_size + 1}: {len(chunk_items)} items")
                            print(f"[INFO] Batch compute_info_gain_batch used {elapsed_time:.2f}s for {len(chunk_items)} items")
                        except Exception as e:
                            print(f"[WARNING] Batch IG service call failed for chunk {chunk_start//chunk_size + 1}: {e}")
                            # Continue with next chunk instead of failing completely
                            # Missing scores will fall back to individual computation in phase 3
                            continue
                except Exception as e:
                    print(f"[WARNING] Batch IG service call failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to individual computation (will happen in phase 3)
                    info_gain_scores_dict = {}
                finally:
                    # Restore original timeout if we modified the global client
                    if ig_client is not None and original_timeout is not None:
                        ig_client.timeout = original_timeout

        # ===== PHASE 3: Compute final scores using pre-computed info gain scores =====
        for i in range(len(data)):
            item_data = item_to_data[i]
            sequences_str = item_data['sequences_str']
            ground_truth = item_data['ground_truth']
            data_source = item_data['data_source']
            compute_score_fn = item_data['compute_score_fn']
            valid_response_length = item_data['valid_response_length']

            # Get pre-computed info gain score if available
            # In validation mode, explicitly set to 0.0 to skip info gain computation
            if is_validation:
                precomputed_info_gain = 0.0
            else:
                precomputed_info_gain = info_gain_scores_dict.get(i, None)

            # Compute score with pre-computed info gain score
            if compute_score_fn.__name__ == 'compute_score_em':
                score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                    format_score=self.format_score,
                    info_gain_score=precomputed_info_gain
                )
                reward_tensor[i, valid_response_length - 1] = score

                # Collect detailed metrics
                metrics = get_last_reward_metrics()
                info_gain_scores.append(metrics['info_gain_score'])
                output_scores.append(metrics['output_score'])  # EM check only
                all_scores.append(metrics['all_score'])  # Weighted sum
            else:
                score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                    format_score=self.format_score
                )
                reward_tensor[i, valid_response_length - 1] = score
                output_scores.append(score)
                all_scores.append(score)
                info_gain_scores.append(0.0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # Store metrics in batch meta_info
        if len(output_scores) > 0:
            if 'metrics' not in data.meta_info:
                data.meta_info['metrics'] = {}
            data.meta_info['metrics'].update({
                'reward/output_score_mean': float(np.mean(output_scores)),  # EM check only
                'reward/output_score_max': float(np.max(output_scores)),
                'reward/output_score_min': float(np.min(output_scores)),
                'reward/all_score_mean': float(np.mean(all_scores)),  # Weighted sum
                'reward/all_score_max': float(np.max(all_scores)),
                'reward/all_score_min': float(np.min(all_scores)),
                'reward/info_gain_score_mean': float(np.mean(info_gain_scores)),
                'reward/info_gain_score_max': float(np.max(info_gain_scores)),
                'reward/info_gain_score_min': float(np.min(info_gain_scores)),
            })

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
