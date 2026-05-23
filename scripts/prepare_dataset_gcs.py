"""
Standalone offline pre-tokenization script for nanoChat.jax.

Downloads FineWeb-Edu parquet dataset shards, iterates through document row groups
with visible tqdm progress tracking, encodes raw strings into tokens using tiktoken,
and persists highly compressed binary token shards (.bin) directly into GCS bucket paths.

Usage:
    python scripts/prepare_dataset_gcs.py --target_dir "/home/iharsh_google_com/iharsh-fuse/dataset_tokens"
"""

import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

from nanochat.dataset import download_single_file, list_parquet_files, parquets_iter_batched
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, print_banner


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Standalone GCS Tokenizer Prep")
    p.add_argument("--target_dir", type=str, default="/home/iharsh_google_com/iharsh-fuse/dataset_tokens", help="Target GCS bucket directory for .bin token shards")
    p.add_argument("--max_shards", type=int, default=1, help="Max raw dataset parquet shards to process")
    p.add_argument("--doc_limit", type=int, default=5000, help="Document count limit for fast initial testing")
    return p.parse_args()


def main():
    print_banner()
    args = parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    print0(f"Targeting cloud storage bucket path: {args.target_dir}")

    # Step 1: Ensure raw parquet file is downloaded
    parquet_files = list_parquet_files()
    if not parquet_files:
        print0("Raw FineWeb-Edu parquet file not found. Downloading shard 0...")
        download_single_file(0)
        parquet_files = list_parquet_files()

    # Step 2: Set up tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    print0(f"Tokenizer initialized successfully (vocab size: {tokenizer.get_vocab_size():,})")

    # Step 3: Stream and encode documents
    tokens_accumulator = []
    total_docs = 0

    print0("\nCommencing offline pre-tokenization processing loop...")
    with tqdm(total=args.doc_limit, desc="Tokenizing FineWeb-Edu Documents", unit="docs") as pbar:
        for batch in parquets_iter_batched(split="val"):
            for text in batch:
                if total_docs >= args.doc_limit:
                    break
                ids = tokenizer.encode(text, prepend=bos_token)
                tokens_accumulator.extend(ids)
                total_docs += 1
                pbar.update(1)
            if total_docs >= args.doc_limit:
                break

    total_tokens = len(tokens_accumulator)
    print0(f"\nProcessing complete: {total_docs:,} documents encoded into {total_tokens:,} tokens.")

    # Step 4: Write binary .bin token shard
    shard_filename = os.path.join(args.target_dir, "dataset_shard_00000.bin")
    
    # Construct binary header: 256 x int32 (magic number 20240520, version 1, total_tokens)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = total_tokens

    print0(f"Writing compressed binary token shard header and tokens to {shard_filename}...")
    with open(shard_filename, "wb") as f:
        f.write(header.tobytes())
        # Convert tokens to uint16 (or uint32 if vocab size exceeds 65535, but grain load expects uint16/int64)
        tokens_arr = np.array(tokens_accumulator, dtype=np.uint16)
        f.write(tokens_arr.tobytes())

    print0(f"Successfully generated pre-tokenized binary shard at: {shard_filename}")
    print0(f"File size: {os.path.getsize(shard_filename) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
