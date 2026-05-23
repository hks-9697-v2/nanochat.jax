"""
Base model pretraining script using JAX, Flax NNX, and Optax.

Utilizes Grain shared-memory loading over pre-tokenized binary (.bin) token shards
persisted in GCS storage buckets. Eliminates runtime tokenization overhead entirely.
Supports distributed multi-axis Mesh sharding (DP, FSDP, TP).

Usage:
    python scripts/base_train.py --shard_dir "/home/iharsh_google_com/iharsh-fuse/dataset_tokens" --num_iterations 100
"""

import argparse
import os
import time
import glob

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.dataloader import pretokenized_distributed_data_loader
from nanochat.common import print0, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import CheckpointManager
from nanochat.loss_eval import evaluate_bpb


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Base Pretraining on GCS Token Shards")
    # Architecture
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    # Optimization
    p.add_argument("--num_iterations", type=int, default=100, help="Training optimization steps")
    p.add_argument("--device_batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--total_batch_size", type=int, default=32768, help="Total effective tokens batch size")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate (AdamW)")
    p.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    # Distributed Parallel Mesh
    p.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    p.add_argument("--fsdp", type=int, default=1, help="FSDP / ZeRO sharding degree")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    # GCS Ingestion and Checkpoint Root
    p.add_argument("--shard_dir", type=str, default="/home/iharsh_google_com/iharsh-fuse/dataset_tokens", help="GCS bucket directory containing pre-tokenized .bin shards")
    p.add_argument("--gcs_bucket", type=str, default="/home/iharsh_google_com/iharsh-fuse/checkpoints", help="GCS bucket directory for model weight checkpoints")
    p.add_argument("--model_tag", type=str, default="gpt2-pretokenized-run", help="Model checkpoint save tag")
    p.add_argument("--eval_every", type=int, default=50, help="Evaluate validation BPB interval")
    return p.parse_args()


def main():
    print_banner()
    args = parse_args()

    devices = jax.devices()
    print0(f"Found {len(devices)} JAX devices: {devices}")
    mesh, data_sharding, param_sharding = setup_distributed_sharding(
        devices, dp=args.dp, fsdp=args.fsdp, tp=args.tp
    )
    print0(f"Distributed mesh initialized with shape (DP={args.dp}, FSDP={args.fsdp}, TP={args.tp})")

    # Locate pre-tokenized shards
    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "*.bin")))
    if not shard_files:
        raise FileNotFoundError(
            f"No binary token shards (.bin) found in {args.shard_dir}. "
            "Please run 'python scripts/prepare_dataset_gcs.py' prior to pretraining."
        )
    print0(f"Located {len(shard_files)} pre-tokenized binary dataset shards in cloud storage.")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocabulary size: {vocab_size:,}")

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
    
    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree.leaves(params))
    print0(f"Initialised GPT base model with {total_params:,} parameters")

    ckpt_dir = os.path.join(args.gcs_bucket, args.model_tag)
    ckpt_manager = CheckpointManager(ckpt_dir, max_to_keep=3)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=max(1, int(args.num_iterations * 0.05)),
        decay_steps=args.num_iterations,
        end_value=args.learning_rate * 0.01,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    with jax.set_mesh(mesh):
        opt_state = nnx.Optimizer(model, opt, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            return model(x, targets=y)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt_state.update(model, grads)
        return loss

    tokens_per_iter = args.device_batch_size * args.max_seq_len
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_iter)
    
    # Initialize high-throughput pre-tokenized loader
    train_loader = pretokenized_distributed_data_loader(
        args.device_batch_size, args.max_seq_len, split="train", shard_files=shard_files
    )

    print0(f"\n--- Commencing Instant Base Pretraining for {args.num_iterations} iterations ---")
    start_time = time.time()
    x_np, y_np = next(train_loader)

    for step in range(args.num_iterations + 1):
        if step == args.num_iterations:
            ckpt_manager.save(step, model=model, extra={"step": step, "params": total_params}, force=True)
            break

        step_start = time.time()
        for _ in range(grad_accum_steps):
            x = jax.device_put(jnp.asarray(x_np, dtype=jnp.int32), data_sharding)
            y = jax.device_put(jnp.asarray(y_np, dtype=jnp.int32), data_sharding)
            loss = train_step(model, opt_state, x, y)
            x_np, y_np = next(train_loader)
        
        jax.block_until_ready(loss)
        step_dt = time.time() - step_start
        tok_per_sec = int((tokens_per_iter * grad_accum_steps) / step_dt) if step_dt > 0 else 0

        if step % 10 == 0 or step == args.num_iterations - 1:
            print0(f"Step {step:05d}/{args.num_iterations:05d} | Loss: {loss.item():.4f} | Throughput: {tok_per_sec:,} tok/s")

    ckpt_manager.close()
    print0(f"Completed pretraining. Total runtime: {(time.time() - start_time) / 60:.2f}m")


if __name__ == "__main__":
    main()
