"""
Base model pretraining script using JAX, Flax NNX, and Optax.

Utilizes Grain shared-memory loading over pre-tokenized binary (.bin) token shards
persisted in cloud storage buckets. Eliminates runtime tokenization overhead entirely.
Supports distributed multi-axis Mesh sharding (DP, FSDP, TP), JAX profiler tracing,
customizable Grain concurrency, and flawless GCS parameter PyTree binding.

Usage:
    python scripts/base_train.py --shard_dir "gs://iharsh-fuse/nano-chat-jax/dataset"
"""

import argparse
import os
import time
import glob
import subprocess

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
    p = argparse.ArgumentParser(description="nanoChat.jax Base Pretraining with GCS Weight Binding")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--num_iterations", type=int, default=100, help="Training optimization steps")
    p.add_argument("--device_batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--total_batch_size", type=int, default=32768, help="Total effective tokens batch size")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate (AdamW)")
    p.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    p.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    p.add_argument("--fsdp", type=int, default=1, help="FSDP / ZeRO sharding degree")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    p.add_argument("--shard_dir", type=str, default="/home/iharsh_google_com/iharsh-fuse/dataset_tokens", help="Cloud storage directory containing pre-tokenized .bin shards")
    p.add_argument("--gcs_bucket", type=str, default="gs://iharsh-fuse/checkpoints/full-dataset-run", help="Direct GCS bucket string for model weight checkpoints")
    p.add_argument("--model_tag", type=str, default="gpt2-pretokenized-run", help="Model checkpoint save tag")
    p.add_argument("--ckpt_every", type=int, default=500, help="Step interval for asynchronous distributed checkpoint saving")
    p.add_argument("--eval_every", type=int, default=50, help="Evaluate validation BPB interval")
    p.add_argument("--grain_workers", type=int, default=4, help="Number of parallel Grain read worker threads")
    p.add_argument("--grain_buffer_size", type=int, default=16, help="Grain map read buffer pool capacity")
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to start JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Step index to start XLA trace recording")
    p.add_argument("--profile_end", type=int, default=-1, help="Step index to stop XLA trace recording")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Trace destination directory")
    return p.parse_args()


def list_shard_files(shard_dir):
    if shard_dir.startswith("gs://"):
        cmd = f"gcloud storage ls '{shard_dir}/*.bin'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            cmd = f"gsutil ls '{shard_dir}/*.bin'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            files = [line.strip() for line in result.stdout.strip().split("\n") if line.strip().endswith(".bin")]
            return sorted(files)
        return []
    else:
        return sorted(glob.glob(os.path.join(shard_dir, "*.bin")))


def sync_to_gcs(local_dir, remote_uri):
    print0(f"Persisting checkpoint directory directly to GCS Bucket: {remote_uri} ...")
    cmd = f"gcloud storage cp --recursive {local_dir} {remote_uri}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        cmd = f"gsutil -m cp -r {local_dir} {remote_uri}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print0(f"Successfully verified permanent model state transfer to {remote_uri}")
    else:
        print0(f"Warning: Transfer hit invariant ({result.stderr})")


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
    print0(f"Found {len(devices)} JAX devices: {devices}")
    mesh, data_sharding, param_sharding = setup_distributed_sharding(
        devices, dp=args.dp, fsdp=args.fsdp, tp=args.tp
    )
    print0(f"Distributed mesh initialized with shape (DP={args.dp}, FSDP={args.fsdp}, TP={args.tp})")

    shard_files = list_shard_files(args.shard_dir)
    if not shard_files:
        raise FileNotFoundError(
            f"No binary token shards (.bin) found in {args.shard_dir}. "
            "Please verify cloud storage URIs and run 'python scripts/prepare_dataset_gcs.py' prior to pretraining."
        )
    print0(f"Located {len(shard_files)} pre-tokenized binary dataset shards across storage root.")

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

    staging_base = f"/tmp/checkpoints/{args.model_tag}"
    os.makedirs(staging_base, exist_ok=True)

    remote_dest = os.path.join(args.gcs_bucket, args.model_tag) if args.gcs_bucket.startswith("gs://") else os.path.join(args.gcs_bucket, args.model_tag)
    if args.gcs_bucket.startswith("gs://"):
        sync_from_gcs(remote_dest, staging_base)

    ckpt_manager = CheckpointManager(staging_base, max_to_keep=3)

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

    start_step = 0
    if ckpt_manager.latest_step() is not None:
        try:
            # Bind the restored parameter module directly into model to completely prevent initial random evaluation
            restored_step, model, restored_opt, extra = ckpt_manager.restore_latest(model=model, opt_state=opt_state)
            start_step = restored_step
            if restored_opt is not None:
                opt_state = restored_opt
            print0(f"Resumed training state from saved checkpoint at step {start_step}.")
        except ValueError as e:
            print0(f"Warning: Restoration invariant encountered ({e}). Commencing pretraining from initial state.")

    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            return model(x, targets=y)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt_state.update(model, grads)
        return loss

    tokens_per_iter = args.device_batch_size * args.max_seq_len
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_iter)
    
    print0(f"Initializing Grain MapDataset loader (Workers={args.grain_workers}, Pre-Buffer={args.grain_buffer_size})...")
    train_loader = pretokenized_distributed_data_loader(
        args.device_batch_size, args.max_seq_len, split="train", shard_files=shard_files,
        grain_workers=args.grain_workers, grain_buffer_size=args.grain_buffer_size
    )

    print0(f"\n--- Commencing Instant Base Pretraining (Steps {start_step} -> {args.num_iterations}) ---")
    start_time = time.time()
    x_np, y_np = next(train_loader)

    for step in range(start_step, args.num_iterations + 1):
        if step == args.profile_start:
            os.makedirs(args.profile_dir, exist_ok=True)
            try:
                jax.profiler.start_trace(args.profile_dir)
                print0(f"\n[Profiler Tracing Enabled] Initiating detailed XLA trace at step {step} to {args.profile_dir}...")
            except Exception as e:
                print0(f"\nWarning: Failed to initiate profiler trace ({e})")

        if step == args.profile_end:
            try:
                jax.profiler.stop_trace()
                print0(f"\n[Profiler Tracing Stopped] XLA trace successfully finalized and saved to {args.profile_dir}.")
            except Exception as e:
                print0(f"\nWarning: Failed to stop profiler trace ({e})")

        if step > start_step and step % args.ckpt_every == 0 and step < args.num_iterations:
            print0(f"\n[Interim Checkpoint Target] Saving training state at step {step}...")
            ckpt_manager.save(step, model=model, extra={"step": step, "params": total_params}, force=True)
            if args.gcs_bucket.startswith("gs://"):
                sync_to_gcs(f"{staging_base}/*", remote_dest)

        if step == args.num_iterations:
            ckpt_manager.save(step, model=model, extra={"step": step, "params": total_params}, force=True)
            if args.gcs_bucket.startswith("gs://"):
                sync_to_gcs(f"{staging_base}/*", remote_dest)
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

        if step % 10 == 0 or step == args.num_iterations - 1 or (args.profile_start <= step < args.profile_end):
            print0(f"Step {step:05d}/{args.num_iterations:05d} | Loss: {loss.item():.4f} | Throughput: {tok_per_sec:,} tok/s")

    ckpt_manager.close()
    print0(f"Completed pretraining. Total runtime: {(time.time() - start_time) / 60:.2f}m")


if __name__ == "__main__":
    main()
