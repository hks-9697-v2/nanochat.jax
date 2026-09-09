"""
nanoChat attention package.
Supports standard JAX einsum attention and hardware-accelerated Tokamax dot product FlashAttention.
"""

from typing import Any
from nanochat.attention.standard import standard_causal_attention
from nanochat.attention.tokamax_dot_product_attention import (
    tokamax_dot_product_attention,
    get_tokamax_tpu_implementation,
)


def dispatch_attention(
    q,
    k,
    v,
    kernel: str = "standard",
    config: Any = None,
    scale: float = 1.0,
    kv_cache: Any = None,
):
    """Unified attention entry point for CausalSelfAttention.

    Args:
        q: Query tensor of shape (B, H, Tq, D)
        k: Key tensor of shape (B, H, Tk, D)
        v: Value tensor of shape (B, H, Tk, D)
        kernel: "standard" or "tokamax"
        config: Model configuration instance (GPTConfig)
        scale: Attention scaling factor
        kv_cache: Optional KVCache instance

    Returns:
        Tensor of shape (B, Tq, H * D)
    """
    if kernel == "tokamax":
        return tokamax_dot_product_attention(q, k, v, config=config, scale=scale, kv_cache=kv_cache)
    elif kernel == "standard":
        return standard_causal_attention(q, k, v, scale=scale, kv_cache=kv_cache)
    else:
        raise ValueError(f"Unknown attention kernel: '{kernel}'. Supported: 'standard', 'tokamax'")


__all__ = [
    "dispatch_attention",
    "standard_causal_attention",
    "tokamax_dot_product_attention",
    "get_tokamax_tpu_implementation",
]
