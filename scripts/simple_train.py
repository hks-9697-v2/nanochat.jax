"""
Train model using JAX + Flax NNX + optax.

Usage:
    python scripts/simple_train.py
    python scripts/simple_train.py --depth 12 --num_iterations 1000
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import print0, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import CheckpointManager
from nanochat.loss_eval import evaluate_bpb

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="nanochat.jax training")
    # Model
    p.add_argument("--depth", type=int, default=20, help="Transformer depth")
    p.add_argument("--max_seq_len", type=int, default=2048, help="Max context length")
    # Training horizon
    p.add_argument("--num_iterations", type=int, default=-1, help="Explicit number of steps (-1 = auto)")
    p.add_argument("--target_param_data_ratio", type=float, default=20, help="Chinchilla-style data:param ratio (-1 = disable)")
    # Optimisation
    p.add_argument("--device_batch_size", type=int, default=32, help="Per-device batch size")
    p.add_argument("--total_batch_size", type=int, default=524288, help="Total batch size in tokens")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate (AdamW)")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (0 = off)")
    # Evaluation
    p.add_argument("--eval_every", type=int, default=250, help="Validate every N steps")
    p.add_argument("--eval_tokens", type=int, default=10485760, help="Tokens for val BPB eval (20*524288)")
    p.add_argument("--sample_every", type=int, default=2000, help="Sample from model every N steps")
    # Output
    p.add_argument("--model_tag", type=str, default="", help="Override checkpoint dir name")
    p.add_argument("--max_to_keep", type=int, default=3, help="Checkpoints to keep")
    return p.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# Derived config
# -----------------------------------------------------------------------------

depth = args.depth
max_seq_len = args.max_seq_len

# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model shape from depth
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers} | model_dim: {model_dim} | num_heads: {num_heads}")

# Batch / gradient-accumulation
tokens_per_fwdbwd = args.device_batch_size * max_seq_len
assert args.total_batch_size % tokens_per_fwdbwd == 0, (
    f"total_batch_size ({args.total_batch_size}) must be divisible by "
    f"device_batch_size*max_seq_len ({tokens_per_fwdbwd})"
)
grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
print0(f"Tokens/micro-batch: {tokens_per_fwdbwd:,} | grad_accum_steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------

model_config = GPTConfig(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
model = GPT(model_config, rngs=nnx.Rngs(0))

# Count parameters
params = nnx.state(model, nnx.Param)
num_params = sum(x.size for x in jax.tree.leaves(params))
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations
num_iterations = args.num_iterations
if num_iterations <= 0 and args.target_param_data_ratio > 0:
    target_tokens = args.target_param_data_ratio * num_params
    num_iterations = int(target_tokens // args.total_batch_size)
    print0(f"Calculated iterations from data:param ratio: {num_iterations:,}")
elif num_iterations <= 0:
    raise ValueError("Specify --num_iterations or --target_param_data_ratio")
else:
    print0(f"Using {num_iterations:,} iterations")

total_tokens = args.total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,} | Tokens:Params ratio: {total_tokens / num_params:.1f}")

# -----------------------------------------------------------------------------
# Optimizer (optax)
# -----------------------------------------------------------------------------

# Warmup + cosine decay schedule
warmup_ratio = 0.01
warmdown_ratio = 0.2
warmup_steps = max(1, round(warmup_ratio * num_iterations))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.learning_rate,
    warmup_steps=warmup_steps,
    decay_steps=num_iterations,
    end_value=args.learning_rate * 0.01,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip) if args.grad_clip > 0 else optax.identity(),
    optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
)

opt_state = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

train_loader = tokenizing_distributed_data_loader(
    args.device_batch_size, max_seq_len, split="train",
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    args.device_batch_size, max_seq_len, split="val",
)

# Kick off the first batch
x_np, y_np = next(train_loader)

# -----------------------------------------------------------------------------
# JIT-compiled training step
# -----------------------------------------------------------------------------

@nnx.jit
def train_step(model, opt_state, x, y):
    """Single forward + backward + optimizer step."""
    def loss_fn(model):
        return model(x, y)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt_state.update(model, grads)
    return loss

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

print0(f"\n{'='*60}")
print0(f"Starting training for {num_iterations} iterations")
print0(f"{'='*60}\n")

min_val_bpb = float("inf")
smooth_train_loss = 0.0
ema_beta = 0.9
total_training_time = 0.0

# Try to load token_bytes for BPB eval (may not exist yet)
try:
    token_bytes = get_token_bytes()
    has_token_bytes = True
except (FileNotFoundError, AssertionError):
    print0("Warning: token_bytes not found — BPB eval will be skipped")
    has_token_bytes = False

# Checkpoint manager
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{depth}"
ckpt_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
ckpt_manager = CheckpointManager(ckpt_dir, max_to_keep=args.max_to_keep)

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # -----------------------------------------------------------------
    # Evaluation: validation BPB
    # -----------------------------------------------------------------
    if has_token_bytes and (last_step or step % args.eval_every == 0):
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * max_seq_len)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    if last_step or (step > 0 and step % args.sample_every == 0):
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
        ]
        rng = jax.random.PRNGKey(step)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            output_tokens = []
            for tok in model.generate(rng, tokens, max_tokens=32, temperature=0.0):
                output_tokens.append(tok)
            full = tokens + output_tokens
            print0(f"  {tokenizer.decode(full)}")

    # -----------------------------------------------------------------
    # Checkpoint at end
    # -----------------------------------------------------------------
    if last_step:
        ckpt_manager.save(
            step,
            model=model,
            extra={
                "step": step,
                "val_bpb": min_val_bpb,
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": vocab_size,
                    "n_layer": num_layers,
                    "n_head": num_heads,
                    "n_kv_head": num_kv_heads,
                    "n_embd": model_dim,
                },
                "num_params": num_params,
                "total_tokens": total_tokens,
            },
            force=True,
        )
        break

    # -----------------------------------------------------------------
    # Training step (with gradient accumulation)
    # -----------------------------------------------------------------
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        x = jnp.asarray(x_np, dtype=jnp.int32)
        y = jnp.asarray(y_np, dtype=jnp.int32)
        train_loss = train_step(model, opt_state, x, y)
        x_np, y_np = next(train_loader)  # prefetch next batch

    # Wait for computation to finish (for accurate timing)
    jax.block_until_ready(train_loss)
    t1 = time.time()
    dt = t1 - t0

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    loss_val = train_loss.item()
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(tokens_per_fwdbwd * grad_accum_steps / dt) if dt > 0 else 0
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt if dt > 0 else 0

    if step > 10:
        total_training_time += dt

    lr_now = float(schedule(step))
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%) | "
        f"loss: {debiased_loss:.4f} | lr: {lr_now:.2e} | "
        f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | "
        f"time: {total_training_time/60:.1f}m"
    )

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

ckpt_manager.close()
print0(f"\nTotal training time: {total_training_time/60:.1f}m")
print0(f"Min validation bpb: {min_val_bpb:.4f}")

# Log to report
try:
    from nanochat.report import get_report
    get_report().log(section="Base model training", data=[
        vars(args),
        {
            "Number of parameters": num_params,
            "FLOPs per token": f"{num_flops_per_token:e}",
            "Iterations": num_iterations,
            "Training tokens": total_tokens,
            "Tokens:Params ratio": total_tokens / num_params,
        },
        {
            "Min validation bpb": min_val_bpb,
            "Total training time": f"{total_training_time/60:.1f}m",
        },
    ])
except Exception as e:
    print0(f"Report logging failed: {e}")