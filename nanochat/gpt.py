"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params using flax.nnx
    return nnx.RMSNorm(num_features=x.shape[-1], use_scale=False, rngs=nnx.Rngs(0))(x)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = jnp.concatenate([y1, y2], axis=3) # re-assemble
    return out.astype(x.dtype) # ensure input/output dtypes match


def repeat_kv(x, n_rep):
    """JAX equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    return jnp.repeat(x, n_rep, axis=1)



class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x, cos_sin, kv_cache=None):
        B, T, C = x.shape

        # Project the input to get queries, keys, and values
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm

        q = jnp.transpose(q, (0, 2, 1, 3)) # (B, H, T, D)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq = q.shape[2] # number of queries in this forward pass
        Tk = k.shape[2] # number of keys/values in total (in the cache + current forward pass)

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k = repeat_kv(k, nrep)
        v = repeat_kv(v, nrep)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        scale = 1.0 / jnp.sqrt(self.head_dim)
        att = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            mask = jnp.tril(jnp.ones((Tq, Tk), dtype=bool))[None, None, :, :]
            att = jnp.where(mask, att, -jnp.inf)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            pass
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            prefix_len = Tk - Tq
            mask = (jnp.arange(Tk)[None, :] <= (jnp.arange(Tq)[:, None] + prefix_len))[None, None, :, :]
            att = jnp.where(mask, att, -jnp.inf)

        att = jax.nn.softmax(att, axis=-1)
        y = jnp.einsum('bhqk,bhkd->bhqd', att, v)

        # Re-assemble the heads side by side and project back to residual stream
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(B, T, -1)
        y = self.c_proj(y)
        return y
    

class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jnp.square(jax.nn.relu(x))
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
    

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.h = [Block(config, layer_idx, rngs=rngs) for layer_idx in range(config.n_layer)]
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)

        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        # self.wte.to(dtype=jnp.bfloat16) # JAX typically handles this via mixed precision policies

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        # autodetect the device from model embeddings (JAX uses default device)
        # stride the channels
        channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = jnp.arange(seq_len, dtype=jnp.float32)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = jnp.outer(t, inv_freq)
        cos, sin = jnp.cos(freqs), jnp.sin(freqs)
        # cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        params = nnx.state(self, nnx.Param)
        nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
        # Exclude non-matmul params: embeddings
        wte_params = nnx.state(self.wte, nnx.Param)
        nparams_exclude = sum(x.size for x in jax.tree_util.tree_leaves(wte_params))
        
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer (assuming full attention, no sliding window in this config)
        attn_flops = 12 * self.config.n_layer * h * q * t
        
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately
        wte_params = nnx.state(self.wte, nnx.Param)
        wte = sum(x.size for x in jax.tree_util.tree_leaves(wte_params))
        
        lm_head_params = nnx.state(self.lm_head, nnx.Param)
        lm_head = sum(x.size for x in jax.tree_util.tree_leaves(lm_head_params))
        
        # transformer matrices (blocks)
        h_params = [nnx.state(block, nnx.Param) for block in self.h]
        transformer_matrices = sum(sum(x.size for x in jax.tree_util.tree_leaves(p)) for p in h_params)
        
        # Total
        params = nnx.state(self, nnx.Param)
        total = sum(x.size for x in jax.tree_util.tree_leaves(params))
        
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'total': total,
        }

    def __call__(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        # assert T <= self.cos.shape[1]
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        
        # truncate cache to current sequence length
        cos = jax.lax.dynamic_slice_in_dim(self.cos, T0, T, axis=1)
        sin = jax.lax.dynamic_slice_in_dim(self.sin, T0, T, axis=1)
        cos_sin = (cos, sin)

        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15.0
        logits = self.lm_head(x)
        logits = softcap * jnp.tanh(logits / softcap) # logits softcap

        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            mask = targets_flat != -1
            safe_targets = jnp.where(mask, targets_flat, 0)
            log_probs = jax.nn.log_softmax(logits_flat)
            target_log_probs = jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)
            loss = -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            return loss
        else:
            # inference mode: compute and return the logits
            return logits

    def generate(self, rng, tokens, max_tokens, temperature=1.0, top_k=None):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        ids = jnp.array([tokens], dtype=jnp.int32) # add batch dim
        for _ in range(max_tokens):
            logits = self(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                top_k_val, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                min_val = top_k_val[:, -1]
                logits = jnp.where(logits < min_val[:, None], -jnp.inf, logits)

            if temperature > 0:
                logits = logits / temperature
                rng, key = jax.random.split(rng)
                next_ids = jax.random.categorical(key, logits, axis=-1)
                next_ids = next_ids[:, None]
            else:
                next_ids = jnp.argmax(logits, axis=-1, keepdims=True)
            
            ids = jnp.concatenate([ids, next_ids], axis=1)
            token = next_ids[0, 0].item()
            yield token