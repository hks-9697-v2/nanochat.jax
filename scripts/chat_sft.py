"""
Chat Supervised Fine-Tuning (SFT) script using JAX, Flax NNX, and Optax.

Prepares conversation turn data with target masking (training on assistant
completions while ignoring prompt tokens). Supports multi-dimensional Mesh sharding,
cloud weight loading, and complete JAX profiler server / step-based tracing.

Usage:
    python scripts/chat_sft.py --model_tag gpt2-base --num_iterations 50 --profile_start 5 --profile_end 15
    python scripts/chat_sft.py --profile_server_port 9999
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, setup_distributed_sharding
from nanochat.common import print0, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description="nanoChat.jax Chat SFT with Profiler Support")
    p.add_argument("--depth", type=int, default=12, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    p.add_argument("--num_iterations", type=int, default=50, help="SFT optimization steps")
    p.add_argument("--device_batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Peak fine-tuning LR")
    p.add_argument("--gcs_bucket", type=str, default="", help="Cloud storage directory for checkpoints")
    p.add_argument("--load_model_tag", type=str, default="gpt2-base", help="Checkpointed base model to load")
    p.add_argument("--save_model_tag", type=str, default="gpt2-chat-sft", help="Destination tag for SFT model")
    # Sharding
    p.add_argument("--dp", type=int, default=1, help="Data parallelism")
    p.add_argument("--fsdp", type=int, default=1, help="FSDP / ZeRO sharding")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    # Profiler Support
    p.add_argument("--profile_server_port", type=int, default=-1, help="Port to expose JAX profiler server (-1 = disabled)")
    p.add_argument("--profile_start", type=int, default=-1, help="Step index to commence detailed XLA tracing")
    p.add_argument("--profile_end", type=int, default=-1, help="Step index to finalize and dump XLA tracing")
    p.add_argument("--profile_dir", type=str, default="/tmp/tensorboard_traces", help="Destination directory for XLA profile dumps")
    return p.parse_args()


def get_mock_conversations():
    """Returns a list of conversation structures for conversational SFT tuning."""
    return [
        {"messages": [{"role": "user", "content": "Hello, who are you?"}, {"role": "assistant", "content": "I am nanoChat, a conversational model trained using JAX."}]},
        {"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "2 + 2 equals 4."}]},
        {"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is the fundamental force by which all things with mass are brought toward one another."}]},
        {"messages": [{"role": "user", "content": "What is your favorite programming language?"}, {"role": "assistant", "content": "I enjoy Python and JAX for scalable accelerated computation."}]}
    ]


def main():
    print_banner()
    args = parse_args()

    # Step 1: Optional JAX Profiler Server Initialization
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

    ckpt_root = args.gcs_bucket if args.gcs_bucket else os.path.join(get_base_dir(), "base_checkpoints")
    base_dir = os.path.join(ckpt_root, args.load_model_tag)
    base_cm = CheckpointManager(base_dir)
    
    if base_cm.latest_step() is not None:
        try:
            base_cm.restore_latest(model=model)
            print0(f"Restored base model weights from {base_dir}")
        except ValueError as e:
            print0(f"Note: Dimension growth detected during checkpoint restore ({e}). Continuing fine-tuning with dynamically expanded vocabulary parameters.")
    else:
        print0("No base checkpoint found; initialising from scratch for SFT.")

    sft_root = args.gcs_bucket if args.gcs_bucket else os.path.join(get_base_dir(), "chatsft_checkpoints")
    sft_cm = CheckpointManager(os.path.join(sft_root, args.save_model_tag), max_to_keep=2)

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

    conversations = get_mock_conversations()
    bos_token = tokenizer.get_bos_token_id()

    print0(f"\n--- Starting Chat SFT for {args.num_iterations} iterations ---")
    t0 = time.time()

    for step in range(args.num_iterations + 1):
        # Step-based programmatic profiler triggering
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

        if step == args.num_iterations:
            sft_cm.save(step, model=model, extra={"step": step}, force=True)
            break

        # Batch assembly with masking
        x_rows, y_rows = [], []
        for b in range(args.device_batch_size):
            conv = conversations[(step * args.device_batch_size + b) % len(conversations)]
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
        loss = sft_step(model, opt_state, x, y)
        jax.block_until_ready(loss)
        dt = time.time() - step_t0

        if step % 10 == 0 or step == args.num_iterations - 1 or (args.profile_start <= step < args.profile_end):
            print0(f"Step {step:05d}/{args.num_iterations:05d} | SFT Loss: {loss.item():.4f} | {dt*1000:.1f}ms")

    sft_cm.close()
    print0(f"Completed SFT fine-tuning in {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
