"""
Base model pretraining script using JAX, Flax NNX, and Optax.

Utilizes Grain shared-memory loading over pre-tokenized binary (.bin) token shards
persisted in cloud storage buckets. Eliminates runtime tokenization overhead entirely.
Supports distributed multi-axis Mesh sharding (DP, FSDP, TP), JAX profiler tracing,
customizable Grain concurrency, flawless GCS parameter binding, ultra-fast targeted single-step syncing,
and animated real-time terminal progress loading spinners.

Usage:
    python scripts/base_train.py --shard_dir "gs://your-bucket-name/nano-chat-jax/dataset"
"""

import argparse
import os
import sys
import time
import glob
import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Apple Silicon jax-metal plugin is currently unstable with Flax NNX PRNG operations.
# Forcing CPU backend for local Mac development to prevent 'default_memory_space is not supported' JaxRuntimeErrors.
if sys.platform == "darwin":
    jax.config.update("jax_platform_name", "cpu")


from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.dataloader import pretokenized_distributed_data_loader
from nanochat.common import print0, print_banner
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import CheckpointManager
from nanochat.loss_eval import evaluate_bpb


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Base Pretraining with Dynamic Loading Spinners")
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
    p.add_argument("--shard_dir", type=str, default="gs://your-bucket-name/dataset_tokens", help="Cloud storage directory containing pre-tokenized .bin shards")
    p.add_argument("--gcs_bucket", type=str, default="gs://your-bucket-name/checkpoints/full-dataset-run", help="Direct GCS bucket string for model weight checkpoints")
    p.add_argument("--checkpoint_dir", type=str, default="/tmp/checkpoints", help="Local directory to stage and save model checkpoints")
    p.add_argument("--model_tag", type=str, default="gpt2-pretokenized-run", help="Model checkpoint save tag")
    p.add_argument("--ckpt_every", type=int, default=500, help="Step interval for asynchronous distributed checkpoint saving")
    p.add_argument("--eval_every", type=int, default=50, help="Evaluate validation BPB interval")
    p.add_argument("--grain_workers", type=int, default=4, help="Number of parallel Grain read worker threads")
    p.add_argument("--grain_buffer_size", type=int, default=16, help="Grain map read buffer pool capacity")
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to start JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Step index to start XLA trace recording")
    p.add_argument("--profile_end", type=int, default=-1, help="Step index to stop XLA trace recording")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Trace destination directory")
    p.add_argument("--attention_kernel", type=str, default="standard", choices=["standard", "tokamax"], help="Attention kernel: 'standard' (einsum) or 'tokamax' (hardware-accelerated flash attention)")
    p.add_argument("--tokamax_tune_mode", type=str, default="auto", choices=["auto", "tokamax_default", "custom"], help="Tokamax tuning mode: 'auto' (tuned defaults), 'tokamax_default' (Tokamax built-in heuristics), or 'custom'")
    p.add_argument("--tokamax_block_q", type=int, default=None, help="Tokamax block_q override")
    p.add_argument("--tokamax_block_kv", type=int, default=None, help="Tokamax block_kv override")
    p.add_argument("--tokamax_block_kv_compute", type=int, default=None, help="Tokamax block_kv_compute override")
    p.add_argument("--tokamax_block_q_dkv", type=int, default=None, help="Tokamax block_q_dkv override")
    p.add_argument("--tokamax_block_kv_dkv", type=int, default=None, help="Tokamax block_kv_dkv override")
    p.add_argument("--tokamax_block_kv_dkv_compute", type=int, default=None, help="Tokamax block_kv_dkv_compute override")
    p.add_argument("--tokamax_q_layout", type=str, default=None, choices=["head_dim_minor", "seq_minor"], help="Tokamax q layout override")
    p.add_argument("--tokamax_k_layout", type=str, default=None, choices=["head_dim_minor", "seq_minor"], help="Tokamax k layout override")
    p.add_argument("--tokamax_v_layout", type=str, default=None, choices=["head_dim_minor", "seq_minor"], help="Tokamax v layout override")
    p.add_argument("--tokamax_use_experimental_scheduler", type=lambda x: (str(x).lower() == 'true'), default=None, help="Tokamax scheduler override (True/False)")
    return p.parse_args()


def run_with_progress(cmd, desc):
    spinner_symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_spinner = False

    def spin():
        i = 0
        while not stop_spinner:
            sys.stdout.write(f"\r{spinner_symbols[i]} {desc} ...")
            sys.stdout.flush()
            i = (i + 1) % len(spinner_symbols)
            time.sleep(0.1)

    t_spin = threading.Thread(target=spin)
    t_spin.start()

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stop_spinner = True
    t_spin.join()

    sys.stdout.write(f"\r[Completed] {desc} \n")
    sys.stdout.flush()
    return proc


def list_shard_files(shard_dir):
    if shard_dir.startswith("gs://"):
        cmd = f"gcloud storage ls '{shard_dir}/*.bin'"
        result = run_with_progress(cmd, f"Discovering remote pre-tokenized binary shards across {shard_dir}")
        if result.returncode != 0:
            cmd = f"gsutil ls '{shard_dir}/*.bin'"
            result = run_with_progress(cmd, f"Discovering binary shards (gsutil fallback)")
        
        if result.returncode == 0 and result.stdout.strip():
            files = [line.strip() for line in result.stdout.strip().split("\n") if line.strip().endswith(".bin")]
            return sorted(files)
        return []
    else:
        return sorted(glob.glob(os.path.join(shard_dir, "*.bin")))


def sync_to_gcs(local_dir, remote_uri):
    cmd = f"gcloud storage cp --recursive {local_dir} {remote_uri}"
    result = run_with_progress(cmd, f"Persisting checkpoint parameters directly to GCS Bucket: {remote_uri}")
    if result.returncode != 0:
        cmd = f"gsutil -m cp -r {local_dir} {remote_uri}"
        run_with_progress(cmd, f"Persisting parameters directly to GCS (gsutil fallback)")


def sync_from_gcs(remote_uri, local_dir):
    if remote_uri.startswith("gs://"):
        os.makedirs(local_dir, exist_ok=True)
        cmd = f"gcloud storage ls '{remote_uri}/'"
        r = run_with_progress(cmd, f"Searching remote GCS checkpointer: {remote_uri} for latest step")
        if r.returncode != 0:
            cmd = f"gsutil ls '{remote_uri}/'"
            r = run_with_progress(cmd, f"Searching remote GCS checkpointer (gsutil fallback)")
        
        latest = -1
        for line in r.stdout.split("\n"):
            clean = line.strip().rstrip("/")
            item = clean.split("/")[-1]
            if item.isdigit():
                latest = max(latest, int(item))
                
        if latest >= 0:
            subprocess.run(f"gcloud storage cp '{remote_uri}/_CHECKPOINT_METADATA' '{local_dir}/' 2>/dev/null", shell=True)
            subprocess.run(f"gsutil cp '{remote_uri}/_CHECKPOINT_METADATA' '{local_dir}/' 2>/dev/null", shell=True)
            
            target_remote = f"{remote_uri}/{latest}"
            cmd = f"gcloud storage cp --recursive '{target_remote}' '{local_dir}/'"
            res = run_with_progress(cmd, f"Pulling checkpoint step {latest} parameters directly from GCS")
            if res.returncode != 0:
                cmd = f"gsutil -m cp -r '{target_remote}' '{local_dir}/'"
                run_with_progress(cmd, f"Pulling checkpoint step {latest} parameters (gsutil fallback)")


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
    num_heads = model_dim // (128 if model_dim % 128 == 0 else 64)
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        attention_kernel=args.attention_kernel,
        tokamax_tune_mode=args.tokamax_tune_mode,
        tokamax_block_q=args.tokamax_block_q,
        tokamax_block_kv=args.tokamax_block_kv,
        tokamax_block_kv_compute=args.tokamax_block_kv_compute,
        tokamax_block_q_dkv=args.tokamax_block_q_dkv,
        tokamax_block_kv_dkv=args.tokamax_block_kv_dkv,
        tokamax_block_kv_dkv_compute=args.tokamax_block_kv_dkv_compute,
        tokamax_q_layout=args.tokamax_q_layout,
        tokamax_k_layout=args.tokamax_k_layout,
        tokamax_v_layout=args.tokamax_v_layout,
        tokamax_use_experimental_scheduler=args.tokamax_use_experimental_scheduler,
        mesh=mesh,
    )

    with jax.set_mesh(mesh):
        model = GPT(config, rngs=nnx.Rngs(0))
    
    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree.leaves(params))
    print0(f"Initialised GPT base model with {total_params:,} parameters")

    staging_base = os.path.join(args.checkpoint_dir, args.model_tag)
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

    global_batch_size = args.device_batch_size * args.dp * args.fsdp
    tokens_per_iter = global_batch_size * args.max_seq_len
    grad_accum_steps = max(1, args.total_batch_size // tokens_per_iter)
    
    print0(f"Initializing Grain MapDataset loader (Workers={args.grain_workers}, Pre-Buffer={args.grain_buffer_size}, GlobalBatch={global_batch_size})...")
    train_loader = pretokenized_distributed_data_loader(
        global_batch_size, args.max_seq_len, split="train", shard_files=shard_files,
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
        with jax.set_mesh(mesh):
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
