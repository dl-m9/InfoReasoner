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
Convert jsonl file(s) to parquet format for training
Supports both single file and batch processing from a directory
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="/forest/forest/Search-R1-info/data/nq/test.jsonl", help='Path to input jsonl file (single file mode)')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to test_data directory (batch mode, scans all subdirectories)')
    parser.add_argument('--output_dir', type=str, default='/forest/forest/Search-R1-info/data/nq', help='Output directory for parquet file')
    parser.add_argument('--data_source', type=str, default='nq', help='Data source identifier (only used in single file mode)')
    parser.add_argument('--template_type', type=str, default='base', help='Template type for prompt')
    parser.add_argument('--split', type=str, default='test', help='Dataset split name')
    parser.add_argument('--hdfs_dir', type=str, default=None, help='Optional HDFS directory to upload')

    args = parser.parse_args()

    all_datasets = []

    # Batch mode: process all jsonl files in test_data directory
    if args.test_data_dir is not None:
        print(f'Batch mode: Scanning {args.test_data_dir} for jsonl files...')
        test_data_dir = os.path.abspath(args.test_data_dir)
        
        # Find all subdirectories
        subdirs = [d for d in os.listdir(test_data_dir) 
                   if os.path.isdir(os.path.join(test_data_dir, d))]
        
        for subdir in sorted(subdirs):
            data_source = subdir  # Use directory name as data_source
            jsonl_pattern = os.path.join(test_data_dir, subdir, f'{args.split}.jsonl')
            
            if os.path.exists(jsonl_pattern):
                print(f'Processing {data_source} from {jsonl_pattern}...')
                dataset = datasets.load_dataset('json', data_files=jsonl_pattern)
                
                # Get the dataset (usually 'train' split from jsonl)
                if 'train' in dataset:
                    split_dataset = dataset['train']
                else:
                    # Use the first available split
                    split_name = list(dataset.keys())[0]
                    split_dataset = dataset[split_name]
                
                # Process the dataset
                def make_map_fn(split, ds_name):
                    def process_fn(example, idx):
                        question_text = example['question'].strip()
                        if question_text[-1] != '?':
                            question_text += '?'
                        
                        # Create a clean example dict for make_prefix
                        clean_example = {'question': question_text}
                        question = make_prefix(clean_example, template_type=args.template_type)
                        solution = {
                            "target": example['golden_answers'],
                        }

                        data = {
                            "data_source": ds_name,
                            "prompt": [{
                                "role": "user",
                                "content": question,
                            }],
                            "ability": "fact-reasoning",
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": solution
                            },
                            "extra_info": {
                                'split': split,
                                'index': idx,
                            }
                        }
                        return data
                    return process_fn

                processed_dataset = split_dataset.map(
                    function=make_map_fn(args.split, data_source), 
                    with_indices=True,
                    remove_columns=split_dataset.column_names  # Remove original columns to avoid schema conflicts
                )
                all_datasets.append(processed_dataset)
                print(f'  Loaded {len(processed_dataset)} examples from {data_source}')
            else:
                print(f'  Warning: {jsonl_pattern} not found, skipping {data_source}')
        
        if len(all_datasets) == 0:
            raise ValueError(f'No jsonl files found in {test_data_dir}')
        
        # Merge all datasets
        print(f'\nMerging {len(all_datasets)} datasets...')
        merged_dataset = datasets.concatenate_datasets(all_datasets)
        processed_dataset = merged_dataset
        
    # Single file mode
    elif args.input_file is not None:
        if args.data_source is None:
            raise ValueError('--data_source must be specified when using --input_file')
        
        data_source = args.data_source
        print(f'Single file mode: Loading jsonl file from {args.input_file}...')
        dataset = datasets.load_dataset('json', data_files=args.input_file)

        # Get the split (usually 'train' or the specified split)
        if args.split in dataset:
            split_dataset = dataset[args.split]
        elif 'train' in dataset:
            print(f'Warning: Split "{args.split}" not found, using "train" instead')
            split_dataset = dataset['train']
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            print(f'Warning: Using first available split "{split_name}"')
            split_dataset = dataset[split_name]

        # Process the dataset
        def make_map_fn(split):
            def process_fn(example, idx):
                question_text = example['question'].strip()
                if question_text[-1] != '?':
                    question_text += '?'
                
                # Create a clean example dict for make_prefix
                clean_example = {'question': question_text}
                question = make_prefix(clean_example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        processed_dataset = split_dataset.map(
            function=make_map_fn(args.split), 
            with_indices=True,
            remove_columns=split_dataset.column_names  # Remove original columns to avoid schema conflicts
        )
    else:
        raise ValueError('Either --input_file or --test_data_dir must be specified')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to parquet
    output_file = os.path.join(args.output_dir, f'{args.split}.parquet')
    print(f'\nSaving parquet file to {output_file}...')
    processed_dataset.to_parquet(output_file)
    print(f'Successfully converted {len(processed_dataset)} examples to parquet format')

    # Optional HDFS upload
    if args.hdfs_dir is not None:
        print(f'Uploading to HDFS: {args.hdfs_dir}...')
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print('Upload completed')

