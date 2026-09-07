"""
Automation script: Download & load Nemotron 3 Ultra dataset, tokenize into binary shards, and run pretraining.

Pipeline:
1. Streams/downloads Nemotron 3 Ultra dataset (default: nvidia/Nemotron-RL-Ultra-Training-Blends) from Hugging Face.
2. Formats documents and tokenizes with tiktoken (BOS delimiter 50256) into compressed binary shards (.bin).
3. Executes distributed pretraining with nanochat's base_train.py utilizing Tokamax FlashAttention.
4. Manages checkpoints on the high-capacity NVMe drive (/mnt/v7x8-disk/iharsh_workspace/).

Usage:
    # Full end-to-end (download, tokenize, and pretrain):
    python scripts/run_nemotron_pretrain.py

    # Prepare data only:
    python scripts/run_nemotron_pretrain.py --prepare_only --max_docs 20000

    # Train only (using previously prepared shards):
    python scripts/run_nemotron_pretrain.py --train_only --num_iterations 500
"""

import argparse
import os
import sys
import subprocess
import numpy as np
from tqdm import tqdm

from nanochat.common import print0, print_banner
from nanochat.tokenizer import get_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Automated Nemotron-3 Ultra Pretraining Pipeline")
    # Dataset & Preprocessing
    p.add_argument(
        "--dataset_name",
        type=str,
        default="nvidia/Nemotron-RL-Ultra-Training-Blends",
        help="Hugging Face dataset identifier for Nemotron",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="reasoning,mopd,ifbench,swe,rlhf",
        help="Comma-separated subset configurations to process from the dataset",
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    p.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/v7x8-disk/iharsh_workspace/data/nemotron_pretrain",
        help="Target directory for pretokenized binary shards (.bin)",
    )
    p.add_argument(
        "--max_docs",
        type=int,
        default=25000,
        help="Maximum document count to process (-1 = unlimited)",
    )
    p.add_argument(
        "--shard_size",
        type=int,
        default=1000000,
        help="Number of tokens per binary shard (.bin)",
    )
    p.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare dataset shards; do not start pretraining",
    )
    p.add_argument(
        "--train_only",
        action="store_true",
        help="Skip data prep and launch pretraining on existing shards",
    )

    # Pretraining hyperparameters
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Context length")
    p.add_argument("--num_iterations", type=int, default=200, help="Pretraining optimization steps")
    p.add_argument("--device_batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--total_batch_size", type=int, default=32768, help="Total effective token batch size")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/mnt/v7x8-disk/iharsh_workspace/checkpoints",
        help="Directory to save checkpoints",
    )
    p.add_argument(
        "--model_tag",
        type=str,
        default="nemotron-base-run",
        help="Tag name for base model checkpoints",
    )
    p.add_argument("--ckpt_every", type=int, default=100, help="Checkpoint interval")
    p.add_argument("--eval_every", type=int, default=50, help="Validation BPB eval interval")

    # Distributed Sharding
    p.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    p.add_argument("--fsdp", type=int, default=8, help="FSDP sharding degree")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")

    # Attention Kernel & Tuning
    p.add_argument(
        "--attention_kernel",
        type=str,
        default="tokamax",
        choices=["standard", "tokamax"],
        help="Attention kernel implementation",
    )
    p.add_argument(
        "--tokamax_tune_mode",
        type=str,
        default="auto",
        choices=["auto", "tokamax_default", "custom"],
        help="Tokamax tuning mode ('auto' = optimal tuned Splash Attention)",
    )
    return p.parse_args()


def extract_raw_text(row):
    """Extracts raw textual training content from various Nemotron schema types."""
    messages = []
    rcp = row.get("responses_create_params") or {}
    if isinstance(rcp, dict) and "input" in rcp and isinstance(rcp["input"], list):
        for msg in rcp["input"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(f"{msg['role']}: {msg['content']}")
    
    ans = row.get("expected_answer") or row.get("answer")
    if ans:
        messages.append(f"assistant: {ans}")
    
    if not messages:
        prompt = row.get("prompt") or row.get("question") or row.get("text") or ""
        if prompt:
            messages.append(str(prompt))
            if ans:
                messages.append(f"assistant: {ans}")

    return "\n\n".join(messages).strip()


def write_shard(filename, tokens_list):
    """Writes pretokenized uint16 tokens to a nanochat .bin shard with 256-int32 header."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic number
    header[1] = 1         # version
    header[2] = len(tokens_list)  # total tokens in shard

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        tokens_arr = np.array(tokens_list, dtype=np.uint16)
        f.write(tokens_arr.tobytes())


def prepare_nemotron_shards(args):
    from datasets import load_dataset

    print0(f"\n[Step 1/2] Downloading & Tokenizing Nemotron Dataset: {args.dataset_name}")
    print0(f"Target shard directory: {args.data_dir}")
    os.makedirs(args.data_dir, exist_ok=True)

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    tokens_buffer = []
    shard_idx = 0
    total_docs = 0
    total_tokens = 0

    pbar = tqdm(total=args.max_docs if args.max_docs > 0 else None, desc="Processing Nemotron Documents", unit="docs")

    for cfg in configs:
        if args.max_docs > 0 and total_docs >= args.max_docs:
            break
        print0(f"\nStreaming config '{cfg}' from {args.dataset_name} ...")
        try:
            ds = load_dataset(args.dataset_name, cfg, split=args.split, streaming=True)
            for row in ds:
                if args.max_docs > 0 and total_docs >= args.max_docs:
                    break
                text = extract_raw_text(row)
                if not text:
                    continue

                ids = tokenizer.encode(text, prepend=bos_token)
                tokens_buffer.extend(ids)
                total_docs += 1
                total_tokens += len(ids)
                pbar.update(1)

                # Flush shard when buffer exceeds target shard_size
                if len(tokens_buffer) >= args.shard_size:
                    shard_file = os.path.join(args.data_dir, f"dataset_shard_{shard_idx:05d}.bin")
                    write_shard(shard_file, tokens_buffer[:args.shard_size])
                    tokens_buffer = tokens_buffer[args.shard_size:]
                    shard_idx += 1

        except Exception as e:
            print0(f"Warning: Failed to process config '{cfg}': {e}")

    pbar.close()

    # Flush remaining tokens into final shard
    if tokens_buffer:
        shard_file = os.path.join(args.data_dir, f"dataset_shard_{shard_idx:05d}.bin")
        write_shard(shard_file, tokens_buffer)
        shard_idx += 1

    print0(f"\nDataset Preparation Completed:")
    print0(f"  Total Documents Processed: {total_docs:,}")
    print0(f"  Total Tokens Encoded:     {total_tokens:,}")
    print0(f"  Total Binary Shards:       {shard_idx}")
    print0(f"  Destination Path:          {args.data_dir}")


def run_pretraining(args):
    print0("\n[Step 2/2] Launching Distributed Pretraining with Tokamax FlashAttention")
    cmd = [
        sys.executable,
        "scripts/base_train.py",
        f"--shard_dir={args.data_dir}",
        f"--checkpoint_dir={args.checkpoint_dir}",
        f"--model_tag={args.model_tag}",
        f"--depth={args.depth}",
        f"--max_seq_len={args.max_seq_len}",
        f"--num_iterations={args.num_iterations}",
        f"--device_batch_size={args.device_batch_size}",
        f"--total_batch_size={args.total_batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--weight_decay={args.weight_decay}",
        f"--attention_kernel={args.attention_kernel}",
        f"--tokamax_tune_mode={args.tokamax_tune_mode}",
        f"--ckpt_every={args.ckpt_every}",
        f"--eval_every={args.eval_every}",
        f"--dp={args.dp}",
        f"--fsdp={args.fsdp}",
        f"--tp={args.tp}",
    ]

    print0(f"Executing: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Pretraining failed with exit code {proc.returncode}")
    print0("\nPretraining completed successfully!")


def main():
    print_banner()
    args = parse_args()

    if not args.train_only:
        prepare_nemotron_shards(args)

    if not args.prepare_only:
        run_pretraining(args)


if __name__ == "__main__":
    main()
