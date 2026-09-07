"""
Chat Supervised Fine-Tuning (SFT) script using JAX, Flax NNX, and Optax.

Loads conversational instruction datasets directly from local files or remote GCS buckets,
applies conversational delimiters, and performs multi-device sharded target-masked training.
Features ultra-fast targeted single-step checkpoint synchronization and real-time animated terminal progress loading spinners.

Usage:
    python scripts/chat_sft.py \
        --gcs_bucket "gs://your-bucket-name/nano-chat-jax/checkpoints/full-dataset-run" \
        --sft_dataset_path "gs://your-bucket-name/nano-chat-jax/sft_dataset/sft_conversations.jsonl" \
        --load_model_tag base_trained_gpt2_full \
        --save_model_tag final_sft_model
"""

import argparse
import os
import sys
import time
import json
import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.common import print0, print_banner
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Chat SFT with Dynamic Loading Spinners")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--num_iterations", type=int, default=500, help="SFT optimization steps")
    p.add_argument("--device_batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Peak fine-tuning LR")
    p.add_argument("--gcs_bucket", type=str, default="gs://your-bucket-name/nano-chat-jax/checkpoints/full-dataset-run", help="GCS bucket for model checkpoints")
    p.add_argument("--checkpoint_dir", type=str, default="/tmp/checkpoints", help="Local directory to stage and save model checkpoints")
    p.add_argument("--sft_dataset_path", type=str, default="gs://your-bucket-name/nano-chat-jax/sft_dataset/sft_conversations.jsonl", help="Path to SFT dataset file (.jsonl)")
    p.add_argument("--load_model_tag", type=str, default="base_trained_gpt2_full", help="Checkpointed base model to load")
    p.add_argument("--save_model_tag", type=str, default="final_sft_model", help="Destination tag for SFT model")
    p.add_argument("--ckpt_every", type=int, default=100, help="Step interval for checkpoint saving")
    p.add_argument("--load_step", type=int, default=-1, help="Explicit step number to restore from checkpoint tag (-1 = latest)")
    # Sharding
    p.add_argument("--dp", type=int, default=1, help="Data parallelism")
    p.add_argument("--fsdp", type=int, default=8, help="FSDP / ZeRO sharding")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    # Profiler Support
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to expose JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Step index to commence detailed XLA tracing")
    p.add_argument("--profile_end", type=int, default=-1, help="Step index to finalize and dump XLA tracing")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Destination directory for XLA profile dumps")
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
    """Executes long network storage operations with a visible real-time progress loading spinner."""
    spinner_symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_spinner = False
    proc_container = {}

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


def load_sft_dataset(path):
    orig_path = str(path)
    if orig_path.startswith("gs://"):
        local_dest = "/tmp/sft_dataset_staging.jsonl"
        cmd = f"gcloud storage cp {orig_path} {local_dest}"
        res = run_with_progress(cmd, f"Syncing remote SFT dataset: {orig_path}")
        if res.returncode != 0:
            cmd = f"gsutil cp {orig_path} {local_dest}"
            run_with_progress(cmd, f"Syncing remote SFT dataset (gsutil fallback)")
        target_file = local_dest
    else:
        target_file = orig_path

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"SFT dataset file not found at: {target_file}")

    print0("Parsing conversations from dataset...")
    conversations = []
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line.strip()))
    
    print0(f"Successfully loaded {len(conversations):,} conversations for supervised fine-tuning.")
    return conversations


def sync_to_gcs(local_dir, remote_uri):
    cmd = f"gcloud storage cp --recursive {local_dir} {remote_uri}"
    res = run_with_progress(cmd, f"Persisting model parameters directly to GCS: {remote_uri}")
    if res.returncode != 0:
        cmd = f"gsutil -m cp -r {local_dir} {remote_uri}"
        run_with_progress(cmd, f"Persisting parameters directly to GCS (gsutil fallback)")


def sync_from_gcs(remote_uri, local_dir, specific_step=-1):
    if remote_uri.startswith("gs://"):
        os.makedirs(local_dir, exist_ok=True)
        if specific_step < 0:
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
                target_step = latest
            else:
                return # No checkpoint steps found
        else:
            target_step = specific_step

        # Synchronize metadata root
        subprocess.run(f"gcloud storage cp '{remote_uri}/_CHECKPOINT_METADATA' '{local_dir}/' 2>/dev/null", shell=True)
        subprocess.run(f"gsutil cp '{remote_uri}/_CHECKPOINT_METADATA' '{local_dir}/' 2>/dev/null", shell=True)
        
        target_remote = f"{remote_uri}/{target_step}"
        cmd = f"gcloud storage cp --recursive '{target_remote}' '{local_dir}/'"
        res = run_with_progress(cmd, f"Pulling checkpoint step {target_step} parameters directly from GCS")
        if res.returncode != 0:
            cmd = f"gsutil -m cp -r '{target_remote}' '{local_dir}/'"
            run_with_progress(cmd, f"Pulling checkpoint step {target_step} parameters (gsutil fallback)")


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
    mesh, data_sharding, param_sharding = setup_distributed_sharding(devices, dp=args.dp, fsdp=args.fsdp, tp=args.tp)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

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

    base_staging = os.path.join(args.checkpoint_dir, args.load_model_tag)
    remote_base = os.path.join(args.gcs_bucket, args.load_model_tag)
    if args.gcs_bucket.startswith("gs://"):
        sync_from_gcs(remote_base, base_staging, specific_step=args.load_step)

    base_cm = CheckpointManager(base_staging)
    
    target_step = args.load_step if args.load_step >= 0 else base_cm.latest_step()
    if target_step is not None:
        try:
            # Bind restored parameter PyTree directly into model
            if args.load_step >= 0:
                _, model, _, _ = base_cm.restore(target_step, model=model)
            else:
                _, model, _, _ = base_cm.restore_latest(model=model)
            print0(f"Restored base model weights directly from {remote_base} step {target_step}")
        except ValueError as e:
            print0(f"Note: Dimension growth detected during checkpoint restore ({e}). Continuing fine-tuning with dynamically expanded vocabulary parameters.")
    else:
        print0("No base checkpoint found; initialising from scratch for SFT.")

    sft_staging = os.path.join(args.checkpoint_dir, args.save_model_tag)
    os.makedirs(sft_staging, exist_ok=True)
    remote_sft = os.path.join(args.gcs_bucket, args.save_model_tag)
    if args.gcs_bucket.startswith("gs://"):
        sync_from_gcs(remote_sft, sft_staging)

    sft_cm = CheckpointManager(sft_staging, max_to_keep=2)

    optimizer = optax.adamw(learning_rate=args.learning_rate)
    with jax.set_mesh(mesh):
        opt_state = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    @nnx.jit
    def sft_step(model, opt_state, x, y):
        def loss_fn(model):
            return model(x, targets=y)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt_state.update(model, grads)
        return loss

    conversations = load_sft_dataset(args.sft_dataset_path)
    bos_token = tokenizer.get_bos_token_id()

    print0(f"\n--- Starting Chat SFT for {args.num_iterations} iterations ---")
    t0 = time.time()

    for step in range(args.num_iterations + 1):
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

        if step > 0 and step % args.ckpt_every == 0 and step < args.num_iterations:
            print0(f"\n[Interim Checkpoint Target] Saving SFT state at step {step}...")
            sft_cm.save(step, model=model, extra={"step": step}, force=True)
            if args.gcs_bucket.startswith("gs://"):
                sync_to_gcs(f"{sft_staging}/*", remote_sft)

        if step == args.num_iterations:
            sft_cm.save(step, model=model, extra={"step": step}, force=True)
            if args.gcs_bucket.startswith("gs://"):
                sync_to_gcs(f"{sft_staging}/*", remote_sft)
            break

        global_batch_size = args.device_batch_size * args.dp * args.fsdp
        x_rows, y_rows = [], []
        for b in range(global_batch_size):
            conv = conversations[(step * global_batch_size + b) % len(conversations)]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=args.max_seq_len + 1)
            
            pad_len = (args.max_seq_len + 1) - len(ids)
            if pad_len > 0:
                ids.extend([bos_token] * pad_len)
                mask.extend([0] * pad_len)
            else:
                ids = ids[:args.max_seq_len + 1]
                mask = mask[:args.max_seq_len + 1]
            
            x_arr = np.array(ids[:-1], dtype=np.int32)
            y_arr = np.array(ids[1:], dtype=np.int32)
            mask_arr = np.array(mask[1:], dtype=np.int32)
            
            y_arr[mask_arr == 0] = -1
            x_rows.append(x_arr)
            y_rows.append(y_arr)

        x = jax.device_put(jnp.array(x_rows, dtype=jnp.int32), data_sharding)
        y = jax.device_put(jnp.array(y_rows, dtype=jnp.int32), data_sharding)

        step_t0 = time.time()
        with jax.set_mesh(mesh):
            loss = sft_step(model, opt_state, x, y)
            jax.block_until_ready(loss)
        dt = time.time() - step_t0

        if step % 10 == 0 or step == args.num_iterations - 1 or (args.profile_start <= step < args.profile_end):
            print0(f"Step {step:05d}/{args.num_iterations:05d} | SFT Loss: {loss.item():.4f} | {dt*1000:.1f}ms")

    sft_cm.close()
    print0(f"Completed SFT fine-tuning in {(time.time() - t0):.2f}s")

    local_dataset_file = "/tmp/sft_dataset_staging.jsonl"
    if os.path.exists(local_dataset_file):
        os.remove(local_dataset_file)


if __name__ == "__main__":
    main()
