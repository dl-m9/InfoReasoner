import argparse
from huggingface_hub import snapshot_download, hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/nq_hotpotqa_train", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, default="data/nq_hotpotqa_train", help="Local directory to save files")
parser.add_argument("--filename", type=str, default=None, help="Specific filename to download (if None, downloads entire repo)")
    
args = parser.parse_args()

if args.filename:
    # Download a specific file
    hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
        local_dir=args.save_path,
    )
else:
    # Download entire repository
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.save_path,
    )
