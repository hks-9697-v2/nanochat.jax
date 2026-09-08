"""
Automation script: Download & load Nemotron conversation dataset, format into SFT dialogues, and run Chat SFT fine-tuning.

Pipeline:
1. Streams/downloads Nemotron dialogue subsets (default: nvidia/Nemotron-RL-Ultra-Training-Blends) from Hugging Face.
2. Extracts and normalizes multi-turn conversations into JSONLines format:
   {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
3. Restores pretrained base model parameters from checkpoint directory.
4. Executes supervised fine-tuning with nanochat's chat_sft.py using Tokamax FlashAttention and ChatML token masking.
5. Saves fine-tuned SFT checkpoints on the high-capacity NVMe drive (/mnt/v7x8-disk/iharsh_workspace/).

Usage:
    # Full end-to-end (download, format, and run SFT):
    python scripts/run_nemotron_sft.py

    # Prepare dataset only:
    python scripts/run_nemotron_sft.py --prepare_only --max_dialogues 10000

    # Train SFT only on existing prepared jsonl:
    python scripts/run_nemotron_sft.py --train_only --num_iterations 300
"""

import argparse
import ast
import glob
import os
import sys
import json
import subprocess
from tqdm import tqdm

from nanochat.common import print0, print_banner


def parse_args():
    p = argparse.ArgumentParser(description="Automated Nemotron SFT Fine-Tuning Pipeline")
    # Dataset & Preprocessing
    p.add_argument(
        "--dataset_name",
        type=str,
        default="nvidia/Nemotron-RL-Ultra-Training-Blends",
        help="Hugging Face dataset identifier for Nemotron SFT data",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="reasoning,rlhf,ifbench,rlvr1",
        help="Comma-separated subset configurations to process",
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    p.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/v7x8-disk/iharsh_workspace/data/nemotron_sft",
        help="Target directory for processed SFT dialogues",
    )
    p.add_argument(
        "--max_dialogues",
        type=int,
        default=25000,
        help="Maximum dialogue count to format (-1 = unlimited)",
    )
    p.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare SFT JSONL dataset; do not launch training",
    )
    p.add_argument(
        "--train_only",
        action="store_true",
        help="Skip data prep and launch SFT on existing JSONL file",
    )

    # SFT Fine-Tuning hyperparameters
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Context length")
    p.add_argument("--num_iterations", type=int, default=300, help="SFT fine-tuning optimization steps")
    p.add_argument("--device_batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Peak fine-tuning learning rate")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/mnt/v7x8-disk/iharsh_workspace/checkpoints",
        help="Directory containing base checkpoints and where SFT checkpoints will be saved",
    )
    p.add_argument(
        "--load_model_tag",
        type=str,
        default="nemotron-base-3b",
        help="Tag name of the base model checkpoint to load weights from",
    )
    p.add_argument(
        "--save_model_tag",
        type=str,
        default="nemotron-sft-run",
        help="Tag name for fine-tuned SFT model checkpoints",
    )
    p.add_argument(
        "--load_step",
        type=int,
        default=-1,
        help="Specific step of the base model checkpoint to restore (-1 = latest)",
    )
    p.add_argument("--ckpt_every", type=int, default=50, help="SFT checkpoint saving interval")
    p.add_argument(
        "--metrics_json_path",
        type=str,
        default="/mnt/v7x8-disk/iharsh_workspace/results/sft_metrics.json",
        help="Destination JSON path for step-by-step SFT performance and loss metrics",
    )

    # Distributed Sharding
    p.add_argument("--dp", type=int, default=8, help="Data parallelism degree")
    p.add_argument("--fsdp", type=int, default=1, help="FSDP sharding degree")
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


def extract_conversation(row):
    """Extracts and normalizes raw Nemotron records into strict alternating user/assistant turns."""
    raw_msgs = []
    rcp = row.get("responses_create_params")
    if isinstance(rcp, str):
        try:
            rcp = ast.literal_eval(rcp)
        except Exception:
            try:
                rcp = json.loads(rcp)
            except Exception:
                rcp = {}
    if not isinstance(rcp, dict):
        rcp = {}

    if "input" in rcp and isinstance(rcp["input"], list):
        for msg in rcp["input"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = str(msg["content"]).strip()
                if content:
                    raw_msgs.append({"role": str(msg["role"]).lower(), "content": content})

    ans = row.get("expected_answer") or row.get("answer")
    if ans:
        ans_str = str(ans).strip()
        if ans_str and (not raw_msgs or raw_msgs[-1]["role"] != "assistant"):
            raw_msgs.append({"role": "assistant", "content": ans_str})

    if not raw_msgs:
        prompt = row.get("prompt") or row.get("question") or ""
        if prompt:
            prompt_str = str(prompt).strip()
            if prompt_str:
                raw_msgs.append({"role": "user", "content": prompt_str})
                if ans:
                    raw_msgs.append({"role": "assistant", "content": str(ans).strip()})

    # Normalize into strictly alternating user/assistant turns starting with user
    merged = []
    system_prefix = ""
    for m in raw_msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system_prefix = (system_prefix + "\n" + content).strip()
        elif role in ("user", "assistant"):
            if not merged:
                if role == "user":
                    if system_prefix:
                        content = f"{system_prefix}\n\n{content}"
                        system_prefix = ""
                    merged.append({"role": "user", "content": content})
                else:
                    # Drop leading assistant message
                    continue
            else:
                if merged[-1]["role"] == role:
                    merged[-1]["content"] += f"\n\n{content}"
                else:
                    merged.append({"role": role, "content": content})

    # Drop trailing user messages if any
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    # Verify alternation and non-empty
    if len(merged) >= 2 and len(merged) % 2 == 0:
        for i, m in enumerate(merged):
            expected = "user" if i % 2 == 0 else "assistant"
            if m["role"] != expected or not isinstance(m["content"], str) or not m["content"].strip():
                return None
        return merged
    return None


def prepare_sft_dataset(args):
    from datasets import load_dataset

    print0(f"\n[Step 1/2] Downloading & Formatting Nemotron SFT Dataset: {args.dataset_name}")
    os.makedirs(args.data_dir, exist_ok=True)
    target_jsonl = os.path.join(args.data_dir, "nemotron_sft.jsonl")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    count = 0

    pbar = tqdm(
        total=args.max_dialogues if args.max_dialogues > 0 else None,
        desc="Extracting Nemotron SFT Dialogues",
        unit="convs",
    )

    with open(target_jsonl, "w", encoding="utf-8") as f_out:
        for cfg in configs:
            if args.max_dialogues > 0 and count >= args.max_dialogues:
                break
            print0(f"\nProcessing SFT config '{cfg}' from {args.dataset_name} ...")
            try:
                from huggingface_hub import hf_hub_download
                fpath = hf_hub_download(args.dataset_name, f"{cfg}.jsonl", repo_type="dataset")
                with open(fpath, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if args.max_dialogues > 0 and count >= args.max_dialogues:
                            break
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            row = json.loads(line_str)
                        except Exception:
                            continue
                        msgs = extract_conversation(row)
                        if not msgs:
                            continue

                        f_out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        count += 1
                        pbar.update(1)

            except Exception as e:
                print0(f"Warning: Failed to process config '{cfg}': {e}")

    pbar.close()
    file_size_mb = os.path.getsize(target_jsonl) / (1024 * 1024)
    print0(f"\nSFT Dataset Preparation Completed:")
    print0(f"  Total Valid Dialogues Extracted: {count:,}")
    print0(f"  Destination JSONL:               {target_jsonl}")
    print0(f"  File Size:                       {file_size_mb:.2f} MB")
    return target_jsonl


def run_sft_training(args, sft_jsonl_path):
    print0("\n[Step 2/2] Launching Distributed SFT Training with Tokamax FlashAttention")
    cmd = [
        sys.executable,
        "scripts/chat_sft.py",
        f"--sft_dataset_path={sft_jsonl_path}",
        f"--checkpoint_dir={args.checkpoint_dir}",
        f"--load_model_tag={args.load_model_tag}",
        f"--save_model_tag={args.save_model_tag}",
        f"--load_step={args.load_step}",
        f"--depth={args.depth}",
        f"--max_seq_len={args.max_seq_len}",
        f"--num_iterations={args.num_iterations}",
        f"--device_batch_size={args.device_batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--attention_kernel={args.attention_kernel}",
        f"--tokamax_tune_mode={args.tokamax_tune_mode}",
        f"--ckpt_every={args.ckpt_every}",
        f"--dp={args.dp}",
        f"--fsdp={args.fsdp}",
        f"--tp={args.tp}",
        f"--metrics_json_path={args.metrics_json_path}",
    ]

    print0(f"Executing: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"SFT fine-tuning failed with exit code {proc.returncode}")
    print0("\nSFT fine-tuning completed successfully!")


def main():
    print_banner()
    args = parse_args()
    target_jsonl = os.path.join(args.data_dir, "nemotron_sft.jsonl")

    if not args.train_only:
        target_jsonl = prepare_sft_dataset(args)
    else:
        if not os.path.exists(target_jsonl):
            raise FileNotFoundError(f"SFT dataset file not found at: {target_jsonl}")

    if not args.prepare_only:
        run_sft_training(args, target_jsonl)


if __name__ == "__main__":
    main()
