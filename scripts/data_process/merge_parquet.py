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
Merge two parquet files into one
"""

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two parquet files')
    parser.add_argument('--input_file1', type=str, default="/forest/forest/Search-R1-info/data/test_data/test.parquet", help='Path to first parquet file')
    parser.add_argument('--input_file2', type=str, default='/forest/forest/Search-R1-info/data/nq_search/test.parquet', help='Path to second parquet file')
    parser.add_argument('--output_file', type=str, default='/forest/forest/Search-R1-info/data/my_test.parquet', help='Path to output merged parquet file')
    parser.add_argument('--hdfs_dir', type=str, default=None, help='Optional HDFS directory to upload')

    args = parser.parse_args()

    # Load both parquet files
    print(f'Loading first file: {args.input_file1}...')
    dataset1 = datasets.load_dataset('parquet', data_files=args.input_file1)
    
    print(f'Loading second file: {args.input_file2}...')
    dataset2 = datasets.load_dataset('parquet', data_files=args.input_file2)

    # Merge all splits from both files
    all_datasets = []
    
    print(f'First file splits: {list(dataset1.keys())}')
    for split_name in dataset1.keys():
        ds = dataset1[split_name]
        all_datasets.append(ds)
        print(f'  Added {split_name}: {len(ds)} examples')
    
    print(f'Second file splits: {list(dataset2.keys())}')
    for split_name in dataset2.keys():
        ds = dataset2[split_name]
        all_datasets.append(ds)
        print(f'  Added {split_name}: {len(ds)} examples')

    print(f'Total datasets to merge: {len(all_datasets)}')

    # Merge all datasets
    print('Merging all datasets...')
    merged_dataset = datasets.concatenate_datasets(all_datasets)
    print(f'Merged dataset: {len(merged_dataset)} examples')

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save merged dataset
    print(f'Saving merged parquet to {args.output_file}...')
    merged_dataset.to_parquet(args.output_file)
    print(f'Successfully merged and saved to {args.output_file}')

    # Optional HDFS upload
    if args.hdfs_dir is not None:
        print(f'Uploading to HDFS: {args.hdfs_dir}...')
        makedirs(args.hdfs_dir)
        copy(src=args.output_file, dst=os.path.join(args.hdfs_dir, os.path.basename(args.output_file)))
        print('Upload completed')

