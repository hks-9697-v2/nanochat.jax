"""
Offline SFT dataset preparation script.

Downloads a high-quality medium subset (25,000 dialogues) of HuggingFaceH4/ultrachat_200k
and saves it directly to Google Cloud Storage as a structured JSONLines file (.jsonl)
for supervised fine-tuning.

Usage:
    python scripts/prepare_sft_gcs.py --gcs_path "gs://iharsh-fuse/nano-chat-jax/sft_dataset/sft_conversations.jsonl"
"""

import argparse
import os
import json
import subprocess
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Download and upload high-quality Ultrachat SFT subset directly to GCS")
    p.add_argument(
        "--gcs_path",
        type=str,
        default="gs://iharsh-fuse/nano-chat-jax/sft_dataset/sft_conversations.jsonl",
        help="Target GCS path for SFT jsonl dataset",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=25000,
        help="Maximum number of SFT dialogues to save",
    )
    return p.parse_args()


def upload_to_gcs(local_file, remote_uri):
    """Direct upload via gcloud storage command execution."""
    print(f"Uploading local SFT dataset to GCS: {remote_uri} ...")
    cmd = f"gcloud storage cp {local_file} {remote_uri}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        cmd = f"gsutil cp {local_file} {remote_uri}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully uploaded SFT dataset to {remote_uri}")
    else:
        print(f"Warning: Upload failed ({result.stderr})")


def main():
    args = parse_args()

    print("Downloading SFT dataset 'HuggingFaceH4/ultrachat_200k' from Hugging Face...")
    # Load train_sft split from HF ultrachat_200k
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    local_dest = "/tmp/sft_conversations.jsonl"
    print(f"Compiling {args.limit} SFT conversations to local cache: {local_dest}...")

    count = 0
    with open(local_dest, "w", encoding="utf-8") as f:
        for item in tqdm(ds, total=args.limit):
            if count >= args.limit:
                break
            
            # Standard SFT formatting containing message dialogues
            messages = item.get("messages")
            if not messages:
                continue
                
            # Strip extra keys to keep the dataset clean and highly readable
            cleaned_messages = []
            for msg in messages:
                cleaned_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
            f.write(json.dumps({"messages": cleaned_messages}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Successfully compiled {count} high-quality conversations locally.")
    
    # Upload completed dataset to GCS bucket
    upload_to_gcs(local_dest, args.gcs_path)
    
    # Clean up local cached jsonl
    if os.path.exists(local_dest):
        os.remove(local_dest)
        print("Cleared local cached files.")


if __name__ == "__main__":
    main()
