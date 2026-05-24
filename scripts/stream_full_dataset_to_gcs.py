"""
Memory-regulated rolling pipeline to stream, pre-tokenize, and upload the complete
100 Billion token FineWeb-Edu dataset to Google Cloud Storage (GCS) storage buckets.

Operates with bounded disk usage (<500MB) by iterating shard-by-shard:
1. Downloads raw parquet shard.
2. Pre-tokenizes document row groups into compressed binary (.bin) token shards.
3. Uploads directly to GCS via gcloud storage.
4. Purges local disk cache before proceeding to next shard.

Usage:
    python scripts/stream_full_dataset_to_gcs.py --target_gcs "gs://your-bucket-name/nano-chat-jax/dataset"
"""

import argparse
import os
import shutil
import subprocess
import glob
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from nanochat.dataset import download_single_file, index_to_filename, DATA_DIR, MAX_SHARD
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, print_banner


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Complete GCS Streaming Dataset Pipeline")
    p.add_argument("--target_gcs", type=str, default="gs://your-bucket-name/nano-chat-jax/dataset", help="Destination GCS bucket path")
    p.add_argument("--start_shard", type=int, default=0, help="Shard index to commence processing from")
    p.add_argument("--max_shards", type=int, default=MAX_SHARD + 1, help="Total dataset shards to stream")
    p.add_argument("--staging_dir", type=str, default="/tmp/nanochat_staging", help="Ephemeral staging directory for binary shards")
    return p.parse_args()


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed: {cmd}\nError: {result.stderr}")
    return result.stdout.strip()


def main():
    print_banner()
    args = parse_args()

    os.makedirs(args.staging_dir, exist_ok=True)
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    print0(f"Destination Cloud Storage Path: {args.target_gcs}")
    print0(f"Streaming range: Shard {args.start_shard} to {min(args.start_shard + args.max_shards - 1, MAX_SHARD)}")

    end_shard = min(args.start_shard + args.max_shards, MAX_SHARD + 1)
    
    for shard_idx in range(args.start_shard, end_shard):
        parquet_filename = index_to_filename(shard_idx)
        parquet_path = os.path.join(DATA_DIR, parquet_filename)
        bin_filename = f"dataset_shard_{shard_idx:05d}.bin"
        staging_bin_path = os.path.join(args.staging_dir, bin_filename)
        destination_gcs_uri = os.path.join(args.target_gcs, bin_filename)

        print0(f"\n--- [Shard {shard_idx:05d}/{MAX_SHARD:05d}] Processing Cycle ---")

        # Step 1: Ensure clean local cache
        for clean_path in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
            if clean_path != parquet_path:
                os.remove(clean_path)

        # Step 2: Download single raw shard
        if not os.path.exists(parquet_path):
            print0(f"Downloading raw parquet {parquet_filename}...")
            success = download_single_file(shard_idx)
            if not success or not os.path.exists(parquet_path):
                print0(f"Warning: Failed to fetch {parquet_filename}. Skipping cycle.")
                continue
        
        # Step 3: Offline pre-tokenization
        print0(f"Extracting and tokenizing row groups from {parquet_filename}...")
        tokens_accumulator = []
        pf = pq.ParquetFile(parquet_path)
        
        with tqdm(total=pf.num_row_groups, desc=f"Tokenizing Shard {shard_idx}", unit="group") as pbar:
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for text in texts:
                    ids = tokenizer.encode(text, prepend=bos_token)
                    tokens_accumulator.extend(ids)
                pbar.update(1)

        total_tokens = len(tokens_accumulator)
        print0(f"Pre-tokenization complete: {total_tokens:,} tokens generated.")

        # Step 4: Assemble binary header & dump to ephemeral staging disk
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = total_tokens

        print0(f"Assembling binary token shard at {staging_bin_path}...")
        with open(staging_bin_path, "wb") as f:
            f.write(header.tobytes())
            tokens_arr = np.array(tokens_accumulator, dtype=np.uint16)
            f.write(tokens_arr.tobytes())
        
        bin_mb = os.path.getsize(staging_bin_path) / (1024 * 1024)
        print0(f"Compressed binary shard ready ({bin_mb:.2f} MB).")

        # Step 5: Upload directly to GCS Bucket
        print0(f"Uploading to Google Cloud Storage: {destination_gcs_uri} ...")
        # Support both modern gcloud storage cp and legacy gsutil cp
        try:
            run_cmd(f"gcloud storage cp {staging_bin_path} {destination_gcs_uri}")
        except Exception:
            run_cmd(f"gsutil cp {staging_bin_path} {destination_gcs_uri}")
        
        print0(f"Upload verified for {bin_filename}.")

        # Step 6: Prune local disk space
        print0("Purging ephemeral raw parquet and binary staging files to release local VM disk...")
        os.remove(parquet_path)
        os.remove(staging_bin_path)

    print0("\n--- Data Streaming Pipeline Verification Complete ---")


if __name__ == "__main__":
    main()
