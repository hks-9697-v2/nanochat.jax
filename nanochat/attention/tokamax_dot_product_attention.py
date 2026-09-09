"""
Hardware-accelerated FlashAttention via Tokamax dot product attention.
Lowers to Pallas / Mosaic TPU Splash Attention on Google Cloud TPUs,
and Triton / CuDNN on NVIDIA GPUs.
"""

from typing import Any
import jax
import jax.numpy as jnp


def _is_tpu_device() -> bool:
    """Checks if any active JAX device is a TPU."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def _parse_qkv_layout(layout_val: Any, default_layout: Any) -> Any:
    """Parses string or enum layout value to Tokamax QKVLayout."""
    if layout_val is None:
        return default_layout
    if isinstance(layout_val, str):
        layout_str = layout_val.lower()
        from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
        if layout_str in ("head_dim_minor", "1"):
            return splash.QKVLayout.HEAD_DIM_MINOR
        elif layout_str in ("seq_minor", "2"):
            return splash.QKVLayout.SEQ_MINOR
    return layout_val


def get_tokamax_tpu_implementation(config: Any, seq_len: int):
    """Returns configured PallasMosaicTpuFlashAttention with tuned or overridden parameters.

    - If tokamax_tune_mode == 'tokamax_default', returns None (using Tokamax's original 128-tile heuristics).
    - If tokamax_tune_mode is 'auto' (default) or 'custom', applies optimal tuned configurations
      discovered via hardware benchmarking on TPU v7x, and allows individual parameter overrides.
    """
    mode = getattr(config, "tokamax_tune_mode", "auto")
    if mode == "tokamax_default":
        return None

    try:
        from tokamax._src.ops.attention import pallas_mosaic_tpu, pallas_mosaic_tpu_vjp
        from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
    except ImportError:
        return None

    # Base tuned defaults from splash_attention_benchmarking on TPU v7x
    if seq_len <= 1024:
        bq = 1024 if seq_len >= 1024 else seq_len
        bkv = 1024 if seq_len >= 1024 else seq_len
        bkv_c = 512 if seq_len >= 512 else seq_len
        bq_dkv = 1024 if seq_len >= 1024 else seq_len
        bkv_dkv = 1024 if seq_len >= 1024 else seq_len
        bkv_dkv_c = 1024 if seq_len >= 1024 else seq_len
        ql = splash.QKVLayout.SEQ_MINOR
        kl = splash.QKVLayout.SEQ_MINOR
        vl = splash.QKVLayout.SEQ_MINOR
        sched = False
    else:
        bq = 512
        bkv = min(seq_len, 2048)
        bkv_c = min(seq_len, 2048)
        bq_dkv = min(seq_len, 1024)
        bkv_dkv = min(seq_len, 2048)
        bkv_dkv_c = min(seq_len, 1024)
        ql = splash.QKVLayout.HEAD_DIM_MINOR
        kl = splash.QKVLayout.HEAD_DIM_MINOR
        vl = splash.QKVLayout.HEAD_DIM_MINOR
        sched = True

    # Apply any explicit config overrides
    if getattr(config, "tokamax_block_q", None) is not None:
        bq = config.tokamax_block_q
    if getattr(config, "tokamax_block_kv", None) is not None:
        bkv = config.tokamax_block_kv
    if getattr(config, "tokamax_block_kv_compute", None) is not None:
        bkv_c = config.tokamax_block_kv_compute
    if getattr(config, "tokamax_block_q_dkv", None) is not None:
        bq_dkv = config.tokamax_block_q_dkv
    if getattr(config, "tokamax_block_kv_dkv", None) is not None:
        bkv_dkv = config.tokamax_block_kv_dkv
    if getattr(config, "tokamax_block_kv_dkv_compute", None) is not None:
        bkv_dkv_c = config.tokamax_block_kv_dkv_compute
    if getattr(config, "tokamax_q_layout", None) is not None:
        ql = _parse_qkv_layout(config.tokamax_q_layout, ql)
    if getattr(config, "tokamax_k_layout", None) is not None:
        kl = _parse_qkv_layout(config.tokamax_k_layout, kl)
    if getattr(config, "tokamax_v_layout", None) is not None:
        vl = _parse_qkv_layout(config.tokamax_v_layout, vl)
    if getattr(config, "tokamax_use_experimental_scheduler", None) is not None:
        sched = config.tokamax_use_experimental_scheduler

    fwd_cfg = pallas_mosaic_tpu.Config(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv_c,
        q_layout=ql,
        k_layout=kl,
        v_layout=vl,
        use_experimental_scheduler=sched,
        use_base2_exp=True,
    )
    bwd_cfg = pallas_mosaic_tpu_vjp.Config(
        block_q_dkv=bq_dkv,
        block_kv_dkv=bkv_dkv,
        block_kv_dkv_compute=bkv_dkv_c,
        use_base2_exp=True,
    )
    vjp = pallas_mosaic_tpu_vjp.PallasMosaicTpuFlashAttentionVjp().replace(config=bwd_cfg)
    return pallas_mosaic_tpu.PallasMosaicTpuFlashAttention(vjp=vjp).replace(config=fwd_cfg)


def _get_distributed_q_sharding(config: Any):
    """Derives q_sharding specification if running across a multi-device distributed mesh."""
    q_shd = getattr(config, "q_sharding", None)
    if q_shd is not None:
        return q_shd

    m = getattr(config, "mesh", None)
    if m is None:
        try:
            abs_m = jax.sharding.get_abstract_mesh()
            if not abs_m.empty:
                m = abs_m
        except Exception:
            m = None

    if m is not None and len(m.axis_names) > 0:
        axis_names = m.axis_names
        if "dp" in axis_names and "fsdp" in axis_names:
            data_axis = ("dp", "fsdp")
        elif "fsdp" in axis_names:
            data_axis = "fsdp"
        elif "dp" in axis_names:
            data_axis = "dp"
        else:
            data_axis = axis_names[0]

        mesh_shape = dict(m.shape)
        num_data_devices = 1
        if isinstance(data_axis, tuple):
            for a in data_axis:
                num_data_devices *= mesh_shape.get(a, 1)
        else:
            num_data_devices = mesh_shape.get(data_axis, 1)

        if num_data_devices > 1:
            return jax.sharding.NamedSharding(
                m, jax.sharding.PartitionSpec(data_axis, None, None, None)
            )
    return None


def tokamax_dot_product_attention(q, k, v, config: Any, scale: float, kv_cache=None):
    """Computes FlashAttention using Tokamax dot_product_attention.

    Args:
        q: Query tensor of shape (B, H, Tq, D)
        k: Key tensor of shape (B, H, Tk, D)
        v: Value tensor of shape (B, H, Tk, D)
        config: GPTConfig instance with tuning options
        scale: Attention scaling factor
        kv_cache: Optional KVCache instance

    Returns:
        Tensor of shape (B, Tq, H * D)
    """
    try:
        import tokamax
        try:
            from absl import flags
            if not flags.FLAGS.is_parsed():
                flags.FLAGS.mark_as_parsed()
        except Exception:
            pass
    except ImportError as e:
        raise ImportError(
            "tokamax is required when attention_kernel='tokamax'. Please install tokamax."
        ) from e

    B, H, Tq, D = q.shape
    Tk = k.shape[2]

    # tokamax.dot_product_attention expects inputs of shape (*B, T, N, H)
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    q_shd = _get_distributed_q_sharding(config)

    extra_kwargs = {}
    if q_shd is not None:
        extra_kwargs["q_sharding"] = q_shd

    if kv_cache is None or Tq == Tk:
        # Full causal attention (training or prompt evaluation)
        if (Tq % 128 == 0) and _is_tpu_device():
            impl = get_tokamax_tpu_implementation(config, Tq)
        else:
            impl = "xla" if (Tq % 128 != 0) else None

        y = tokamax.dot_product_attention(
            q_t, k_t, v_t, scale=scale, is_causal=True, implementation=impl, **extra_kwargs
        )
    elif Tq == 1:
        # Single-token autoregressive generation against past KV cache
        y = tokamax.dot_product_attention(
            q_t, k_t, v_t, scale=scale, is_causal=False, implementation="xla"
        )
    else:
        # Chunked generation
        prefix_len = Tk - Tq
        mask = (jnp.arange(Tk)[None, :] <= (jnp.arange(Tq)[:, None] + prefix_len))[None, None, :, :]
        y = tokamax.dot_product_attention(
            q_t, k_t, v_t, mask=mask, scale=scale, implementation="xla"
        )

    return y.reshape(B, Tq, -1)
