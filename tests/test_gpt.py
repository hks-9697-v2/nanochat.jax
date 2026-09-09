"""
Unit tests for nanoChat.jax GPT architecture, KVCache, and distributed helpers.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

try:
    from absl import flags
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
except Exception:
    pass

from nanochat.gpt import GPT, GPTConfig, KVCache, setup_distributed_sharding


def test_gpt_initialization_and_forward():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    
    x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    logits = model(x)
    assert logits.shape == (2, 4, 1024)


def test_gpt_loss_computation():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    
    x = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    targets = jnp.array([[2, 3, 4, -1]], dtype=jnp.int32)
    
    loss = model(x, targets=targets)
    assert loss.ndim == 0
    assert not jnp.isnan(loss)


def test_kv_cache_mechanism():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    cache = KVCache(config, max_batch_size=1)
    
    assert cache.get_pos() == 0
    head_dim = config.n_embd // config.n_head
    k = jnp.ones((1, config.n_kv_head, 4, head_dim))
    v = jnp.ones((1, config.n_kv_head, 4, head_dim))
    
    ret_k, ret_v = cache.insert_kv(layer_idx=0, k=k, v=v)
    assert ret_k.shape == (1, config.n_kv_head, 4, head_dim)
    
    cache.update_pos(4)
    assert cache.get_pos() == 4


def test_setup_distributed_sharding():
    devices = jax.devices()
    mesh, data_sharding, param_sharding = setup_distributed_sharding(devices, dp=1, fsdp=1, tp=1)
    assert mesh is not None
    assert data_sharding is not None
    assert param_sharding is not None


def test_tokamax_attention_forward():
    import pytest
    pytest.importorskip("tokamax")

    # Test with sequence_len=128 (TPU block-size aligned for flash attention)
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 128), dtype=jnp.int32)
    logits = model(x)
    assert logits.shape == (2, 128, 1024)
    assert not jnp.any(jnp.isnan(logits))

    # Test with short sequence length (T=32) fallback
    config_short = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
    )
    model_short = GPT(config_short, rngs=nnx.Rngs(0))
    x_short = jnp.ones((2, 32), dtype=jnp.int32)
    logits_short = model_short(x_short)
    assert logits_short.shape == (2, 32, 1024)
    assert not jnp.any(jnp.isnan(logits_short))


def test_tokamax_and_standard_attention_equivalence():
    import pytest
    pytest.importorskip("tokamax")

    # Config with T=128
    config_std = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2, # test MQA / GQA equivalence as well
        n_embd=256,
        attention_kernel="standard",
    )
    config_tok = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=256,
        attention_kernel="tokamax",
    )

    rngs = nnx.Rngs(42)
    model_std = GPT(config_std, rngs=rngs)
    # Copy exact parameter state from standard model into tokamax model
    graphdef, state = nnx.split(model_std)
    model_tok = GPT(config_tok, rngs=nnx.Rngs(42))
    nnx.update(model_tok, state)

    # Random integer tokens
    key = jax.random.PRNGKey(123)
    x = jax.random.randint(key, (2, 128), minval=0, maxval=1024, dtype=jnp.int32)

    logits_std = model_std(x)
    logits_tok = model_tok(x)

    assert logits_std.shape == logits_tok.shape
    max_diff = float(jnp.max(jnp.abs(logits_std - logits_tok)))
    # Flash attention reorders operations and uses different summation trees;
    # max difference should be within small numerical tolerance (< 0.05).
    assert max_diff < 0.05, f"Logits differ too much: max_diff = {max_diff}"

    # Also check equivalence with short sequence length (T=32)
    config_std_32 = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="standard",
    )
    config_tok_32 = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
    )
    model_std_32 = GPT(config_std_32, rngs=nnx.Rngs(0))
    _, state_32 = nnx.split(model_std_32)
    model_tok_32 = GPT(config_tok_32, rngs=nnx.Rngs(0))
    nnx.update(model_tok_32, state_32)

    x_32 = jax.random.randint(key, (2, 32), minval=0, maxval=1024, dtype=jnp.int32)
    logits_std_32 = model_std_32(x_32)
    logits_tok_32 = model_tok_32(x_32)
    max_diff_32 = float(jnp.max(jnp.abs(logits_std_32 - logits_tok_32)))
    assert max_diff_32 < 1e-4, f"Short sequence logits differ: max_diff = {max_diff_32}"


def test_tokamax_tuning_options():
    import pytest
    pytest.importorskip("tokamax")

    # 1. Test auto mode (tuned defaults)
    cfg_auto = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
        tokamax_tune_mode="auto",
    )
    model_auto = GPT(cfg_auto, rngs=nnx.Rngs(1))
    x = jnp.ones((2, 128), dtype=jnp.int32)
    out_auto = model_auto(x)
    assert out_auto.shape == (2, 128, 1024)
    assert not jnp.any(jnp.isnan(out_auto))

    # 2. Test tokamax_default mode (built-in 128-block heuristics)
    cfg_default = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
        tokamax_tune_mode="tokamax_default",
    )
    model_default = GPT(cfg_default, rngs=nnx.Rngs(1))
    out_default = model_default(x)
    assert out_default.shape == (2, 128, 1024)
    assert not jnp.any(jnp.isnan(out_default))

    # 3. Test custom overrides
    cfg_custom = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        attention_kernel="tokamax",
        tokamax_tune_mode="custom",
        tokamax_block_q=128,
        tokamax_block_kv=128,
        tokamax_block_kv_compute=128,
        tokamax_block_q_dkv=128,
        tokamax_block_kv_dkv=128,
        tokamax_block_kv_dkv_compute=128,
        tokamax_q_layout="seq_minor",
        tokamax_use_experimental_scheduler=False,
    )
    model_custom = GPT(cfg_custom, rngs=nnx.Rngs(1))
    out_custom = model_custom(x)
    assert out_custom.shape == (2, 128, 1024)
    assert not jnp.any(jnp.isnan(out_custom))


def test_kernel_dispatcher_direct():
    from nanochat.attention import dispatch_attention, standard_causal_attention, tokamax_dot_product_attention
    q = jnp.ones((1, 2, 8, 16))
    k = jnp.ones((1, 2, 8, 16))
    v = jnp.ones((1, 2, 8, 16))
    out = dispatch_attention(q, k, v, kernel="standard", scale=0.25)
    assert out.shape == (1, 8, 32)


def test_num_scaling_params():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1024,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    counts = model.num_scaling_params()
    assert "total" in counts
    assert "transformer_matrices" in counts
    assert "wte" in counts
    assert "lm_head" in counts
    assert counts["total"] > 0
    assert counts["total"] == counts["transformer_matrices"] + counts["wte"] + counts["lm_head"]

