"""
Unit tests for nanoChat.jax GPT architecture, KVCache, and distributed helpers.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

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
