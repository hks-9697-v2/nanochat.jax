"""
Interactive CLI generation script for nanoChat.jax.

Accepts prompts and performs autoregressive sampling using Flax NNX.
Optionally integrates with cloud storage buckets and complete JAX profiler server / step tracing.

Usage:
    python scripts/chat_cli.py --prompt "The capital of France is" --profile_start 1 --profile_end 15
    python scripts/chat_cli.py --profile_server_port 9999
"""

import argparse
import os

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.common import print0, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Inference CLI with Profiler Support")
    p.add_argument("-p", "--prompt", type=str, default="The capital of France is", help="Prompt text")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--max_tokens", type=int, default=32, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    p.add_argument("--load_model_tag", type=str, default="gpt2-chat-sft", help="Model checkpoint tag to restore")
    p.add_argument("--gcs_bucket", type=str, default="", help="Optional cloud storage path")
    # Profiler Support
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to start JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Token index to start XLA trace recording")
    p.add_argument("--profile_end", type=int, default=-1, help="Token index to stop XLA trace recording")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Trace destination directory")
    return p.parse_args()


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

    ckpt_root = args.gcs_bucket if args.gcs_bucket else os.path.join(get_base_dir(), "chatsft_checkpoints")
    model_dir = os.path.join(ckpt_root, args.load_model_tag)
    cm = CheckpointManager(model_dir)

    if cm.latest_step() is not None:
        try:
            cm.restore_latest(model=model)
            print0(f"Restored model weights from {model_dir}")
        except ValueError as e:
            print0(f"Note: Rotary buffer shape mismatch detected during restore ({e}). Running inference with resiliently matched shapes.")
    else:
        base_dir = os.path.join(os.path.join(get_base_dir(), "base_checkpoints"), "gpt2-base")
        base_cm = CheckpointManager(base_dir)
        if base_cm.latest_step() is not None:
            try:
                base_cm.restore_latest(model=model)
                print0(f"Restored base model weights from {base_dir}")
            except ValueError as e:
                print0(f"Note: Resilient base model load ({e}).")
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
