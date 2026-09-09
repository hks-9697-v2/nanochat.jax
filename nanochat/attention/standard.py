"""
Standard causal attention implementation using pure JAX einsum with causal masking.
Compatible across CPU, GPU, and TPU architectures without external dependencies.
"""

import jax
import jax.numpy as jnp


def standard_causal_attention(q, k, v, scale: float, kv_cache=None):
    """Computes autoregressive causal multi-head attention via einsum and softmax.

    Args:
        q: Query tensor of shape (B, H, Tq, D)
        k: Key tensor of shape (B, H, Tk, D)
        v: Value tensor of shape (B, H, Tk, D)
        scale: Attention scaling factor (typically 1.0 / sqrt(head_dim))
        kv_cache: Optional KVCache instance

    Returns:
        Tensor of shape (B, Tq, H * D)
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[2]

    # Attention: queries attend to keys/values autoregressively
    att = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if kv_cache is None or Tq == Tk:
        # Full causal attention (training or prompt evaluation without cache)
        mask = jnp.tril(jnp.ones((Tq, Tk), dtype=bool))[None, None, :, :]
        att = jnp.where(mask, att, -jnp.inf)
    elif Tq == 1:
        # Single-token autoregressive generation against past cached keys/values
        pass
    else:
        # Chunked generation with existing prefix in cache
        prefix_len = Tk - Tq
        mask = (jnp.arange(Tk)[None, :] <= (jnp.arange(Tq)[:, None] + prefix_len))[None, None, :, :]
        att = jnp.where(mask, att, -jnp.inf)

    att = jax.nn.softmax(att, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", att, v)

    # Re-assemble the heads side by side
    y = jnp.transpose(y, (0, 2, 1, 3)).reshape(B, Tq, -1)
    return y
