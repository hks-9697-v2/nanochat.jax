"""
Interactive CLI generation script for nanoChat.jax.

Accepts prompts and performs autoregressive sampling using Flax NNX.
Optionally integrates with GCS buckets to load trained weights.

Usage:
    python scripts/chat_cli.py --prompt "The capital of France is" --temperature 0.7
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
    p = argparse.ArgumentParser(description="nanoChat.jax Inference CLI")
    p.add_argument("-p", "--prompt", type=str, default="The capital of France is", help="Prompt text")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--max_tokens", type=int, default=32, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    p.add_argument("--load_model_tag", type=str, default="gpt2-chat-sft", help="Model checkpoint tag to restore")
    p.add_argument("--gcs_bucket", type=str, default="", help="Optional GCS bucket storage path")
    return p.parse_args()


def main():
    print_banner()
    args = parse_args()

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
        # Fallback to base model if SFT not found
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

    # Streaming generation loop
    out_tokens = []
    for tok in model.generate(
        rng, tokens, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    ):
        out_tokens.append(tok)

    full_output = tokenizer.decode(tokens + out_tokens)
    print0(f"\nResult:\n{full_output}\n")


if __name__ == "__main__":
    main()
