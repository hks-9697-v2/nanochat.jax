"""
A number of functions that help with evaluating a base model.
"""
import math

import jax
import jax.numpy as jnp
from flax import nnx


def per_token_cross_entropy(logits, targets):
    """Compute per-token cross-entropy loss (no reduction).

    Parameters
    ----------
    logits : jnp.ndarray, shape (B, T, V)
    targets : jnp.ndarray, shape (B, T), int32/int64

    Returns
    -------
    loss : jnp.ndarray, shape (B*T,), per-token nats (unreduced).
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])  # (B*T, V)
    targets_flat = targets.reshape(-1)                   # (B*T,)
    log_probs = jax.nn.log_softmax(logits_flat)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets_flat[:, None], axis=-1
    ).squeeze(-1)
    return -target_log_probs  # (B*T,)  positive nats


def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-indepedent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes array:
    It is a 1D array of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).

    Parameters
    ----------
    model : GPT (nnx.Module)
        The model in eval mode (no dropout etc.).
    batches : iterable
        Yields (inputs, targets) as numpy or jax arrays.
    steps : int
        Number of batches to evaluate.
    token_bytes : jnp.ndarray, shape (vocab_size,)
        Byte length per token (0 = skip that token).

    Returns
    -------
    bpb : float
        Bits per byte.
    """
    token_bytes = jnp.asarray(token_bytes)
    total_nats = 0.0
    total_bytes = 0

    @nnx.jit
    def eval_step(model, x, y):
        # Forward pass — get logits (no targets → inference mode)
        logits = model(x)                     # (B, T, V)
        return per_token_cross_entropy(logits, y)  # (B*T,)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        x, y = jnp.asarray(x), jnp.asarray(y)

        loss2d = eval_step(model, x, y)

        y_flat = y.reshape(-1)

        # Handle masked targets (< 0 means ignore)
        valid = y_flat >= 0
        y_safe = jnp.where(valid, y_flat, 0)

        # Map valid targets to their byte length; ignored/special tokens contribute 0
        num_bytes_flat = jnp.where(valid, token_bytes[y_safe], 0)
        countable = num_bytes_flat > 0

        total_nats += jnp.sum(loss2d * countable).item()
        total_bytes += jnp.sum(num_bytes_flat).item()

    # Multi-process reduction (for future multi-host JAX)
    if jax.process_count() > 1:
        total_nats = jax.experimental.multihost_utils.process_allgather(
            jnp.array(total_nats)
        ).sum().item()
        total_bytes = jax.experimental.multihost_utils.process_allgather(
            jnp.array(total_bytes)
        ).sum().item()

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb