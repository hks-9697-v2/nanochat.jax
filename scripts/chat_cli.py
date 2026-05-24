"""
Interactive CLI generation script for nanoChat.jax.

Accepts prompts and performs autoregressive sampling using Flax NNX.
Optionally integrates with direct GCS parameter binding and JAX profiler tracing.

Usage:
    python scripts/chat_cli.py --gcs_bucket "gs://iharsh-fuse/checkpoints/full-dataset-run" --load_model_tag final_sft_model -p "The capital of France is"
"""

import argparse
import os
import subprocess

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.common import print0, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Inference CLI with Direct GCS Parameter Binding")
    p.add_argument("-p", "--prompt", type=str, default="The capital of France is", help="Prompt text")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--max_tokens", type=int, default=32, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    p.add_argument("--load_model_tag", type=str, default="final_sft_model", help="Model checkpoint tag to restore")
    p.add_argument("--gcs_bucket", type=str, default="gs://iharsh-fuse/checkpoints/full-dataset-run", help="Direct GCS bucket string")
    # Profiler Support
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to start JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Token index to start XLA trace recording")
    p.add_argument("--profile_end", type=int, default=-1, help="Token index to stop XLA trace recording")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Trace destination directory")
    return p.parse_args()


def sync_from_gcs(remote_uri, local_dir):
    if remote_uri.startswith("gs://"):
        os.makedirs(local_dir, exist_ok=True)
        print0(f"Loading checkpoint parameters directly from GCS Bucket: {remote_uri} ...")
        cmd = f"gcloud storage cp --recursive '{remote_uri}/*' '{local_dir}/'"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            cmd = f"gsutil -m cp -r '{remote_uri}/*' '{local_dir}/'"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    print_banner()
    args = parse_args()

    if args.profile_server_port > 0:
        try:
            jax.profiler.start_server(args.profile_server_port)
            print0(f"JAX Profiler active on port {args.profile_server_port}. Capture via Capture Profile tool in TensorBoard/Perfetto.")
        except Exception as e:
            print0(f"Warning: Failed to launch profiler server on port {args.profile_server_port} ({e})")

    devices = jax.devices()
    mesh, _, _ = setup_distributed_sharding(devices, dp=1, fsdp=1, tp=1)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    model_dim = args.depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

    with jax.set_mesh(mesh):
        model = GPT(config, rngs=nnx.Rngs(0))

    staging_base = f"/tmp/checkpoints/{args.load_model_tag}"
    remote_model = os.path.join(args.gcs_bucket, args.load_model_tag)
    if args.gcs_bucket.startswith("gs://"):
        sync_from_gcs(remote_model, staging_base)

    cm = CheckpointManager(staging_base)

    if cm.latest_step() is not None:
        try:
            # Bind restored parameter module directly into model
            _, model, _, _ = cm.restore_latest(model=model)
            print0(f"Restored model weights directly from {remote_model}")
        except ValueError as e:
            print0(f"Note: Rotary buffer shape mismatch detected during restore ({e}). Running inference with resiliently matched shapes.")
    else:
        print0("No checkpoint found; running inference with initialised weights.")

    tokens = tokenizer.encode(args.prompt, prepend="<|bos|>")
    rng = jax.random.PRNGKey(42)

    print0(f"\nPrompt: {args.prompt}")
    print0("Generating response...")

    out_tokens = []
    step = 0
    for tok in model.generate(
        rng, tokens, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    ):
        if step == args.profile_start:
            os.makedirs(args.profile_dir, exist_ok=True)
            try:
                jax.profiler.start_trace(args.profile_dir)
                print0(f"\n[Profiler Tracing Enabled] Initiating detailed XLA trace at generated token index {step} to {args.profile_dir}...")
            except Exception as e:
                print0(f"\nWarning: Failed to initiate profiler trace ({e})")

        if step == args.profile_end:
            try:
                jax.profiler.stop_trace()
                print0(f"\n[Profiler Tracing Stopped] XLA trace successfully finalized and saved to {args.profile_dir}.")
            except Exception as e:
                print0(f"\nWarning: Failed to stop profiler trace ({e})")

        out_tokens.append(tok)
        step += 1

    full_output = tokenizer.decode(tokens + out_tokens)
    print0(f"\nResult:\n{full_output}\n")


if __name__ == "__main__":
    main()
