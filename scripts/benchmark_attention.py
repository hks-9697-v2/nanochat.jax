"""
Performance comparison benchmark: Standard Attention vs. Tokamax FlashAttention.
Measures:
1. Isolated attention forward and backward latency across sequence lengths.
2. End-to-end full model training step latency and throughput (tokens/sec).
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanochat.gpt import GPT, GPTConfig, CausalSelfAttention, get_tokamax_tpu_implementation


def benchmark_isolated_attention(seq_lengths=(1024, 2048, 4096, 8192), batch_size=4, n_head=6, head_dim=64, warmup=5, iters=20):
    print("\n" + "=" * 80)
    print("1. ISOLATED ATTENTION BENCHMARK (Forward & Backward)")
    print(f"Hardware: {jax.devices()[0].device_kind} | Batch: {batch_size}, Heads: {n_head}, Head Dim: {head_dim}")
    print("=" * 80)

    try:
        import tokamax
        from absl import flags
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
    except Exception:
        pass

    results = []

    for T in seq_lengths:
        print(f"\nEvaluating Sequence Length T = {T} ...")

        # Standard attention fwd & bwd
        def std_fwd_bwd():
            scale = float(1.0 / (head_dim ** 0.5))

            def loss_fn(q, k, v):
                att = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
                mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
                att = jnp.where(mask, att, -jnp.inf)
                att = jax.nn.softmax(att, axis=-1)
                out = jnp.einsum('bhqk,bhkd->bhqd', att, v)
                return jnp.sum(out)

            grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))
            return grad_fn

        # Tokamax attention fwd & bwd (untuned default heuristic)
        def tok_default_fwd_bwd():
            scale = float(1.0 / (head_dim ** 0.5))

            def loss_fn(q_t, k_t, v_t):
                out = tokamax.dot_product_attention(
                    q_t, k_t, v_t, scale=scale, is_causal=True, implementation="mosaic_tpu"
                )
                return jnp.sum(out)

            grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))
            return grad_fn

        # Tokamax attention fwd & bwd (tuned Splash attention)
        def tok_tuned_fwd_bwd():
            scale = float(1.0 / (head_dim ** 0.5))
            cfg = GPTConfig(sequence_len=T, attention_kernel="tokamax", tokamax_tune_mode="auto")
            impl = get_tokamax_tpu_implementation(cfg, T)

            def loss_fn(q_t, k_t, v_t):
                out = tokamax.dot_product_attention(
                    q_t, k_t, v_t, scale=scale, is_causal=True, implementation=impl
                )
                return jnp.sum(out)

            grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))
            return grad_fn

        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        # Standard inputs: (B, H, T, D) in bfloat16
        q_std = jax.random.normal(k1, (batch_size, n_head, T, head_dim), dtype=jnp.bfloat16)
        k_std = jax.random.normal(k2, (batch_size, n_head, T, head_dim), dtype=jnp.bfloat16)
        v_std = jax.random.normal(k3, (batch_size, n_head, T, head_dim), dtype=jnp.bfloat16)

        # Tokamax inputs: (B, T, H, D) in bfloat16
        q_tok = jnp.transpose(q_std, (0, 2, 1, 3))
        k_tok = jnp.transpose(k_std, (0, 2, 1, 3))
        v_tok = jnp.transpose(v_std, (0, 2, 1, 3))

        # Bench Standard
        std_time_ms = None
        try:
            fn_std = std_fwd_bwd()
            for _ in range(warmup):
                val, grads = fn_std(q_std, k_std, v_std)
                jax.block_until_ready(grads)

            t0 = time.perf_counter()
            for _ in range(iters):
                val, grads = fn_std(q_std, k_std, v_std)
                jax.block_until_ready(grads)
            t1 = time.perf_counter()
            std_time_ms = (t1 - t0) / iters * 1000.0
        except Exception as e:
            print(f"  Standard Attention failed/OOM at T={T}: {e}")

        # Bench Tokamax Default Heuristic
        tok_default_ms = None
        try:
            fn_tok_def = tok_default_fwd_bwd()
            for _ in range(warmup):
                val, grads = fn_tok_def(q_tok, k_tok, v_tok)
                jax.block_until_ready(grads)

            t0 = time.perf_counter()
            for _ in range(iters):
                val, grads = fn_tok_def(q_tok, k_tok, v_tok)
                jax.block_until_ready(grads)
            t1 = time.perf_counter()
            tok_default_ms = (t1 - t0) / iters * 1000.0
        except Exception as e:
            print(f"  Tokamax (Default) Attention failed at T={T}: {e}")

        # Bench Tokamax Tuned Splash
        tok_tuned_ms = None
        try:
            fn_tok_tuned = tok_tuned_fwd_bwd()
            for _ in range(warmup):
                val, grads = fn_tok_tuned(q_tok, k_tok, v_tok)
                jax.block_until_ready(grads)

            t0 = time.perf_counter()
            for _ in range(iters):
                val, grads = fn_tok_tuned(q_tok, k_tok, v_tok)
                jax.block_until_ready(grads)
            t1 = time.perf_counter()
            tok_tuned_ms = (t1 - t0) / iters * 1000.0
        except Exception as e:
            print(f"  Tokamax (Tuned) Attention failed at T={T}: {e}")

        speedup_vs_std = f"{std_time_ms / tok_tuned_ms:.2f}x" if (std_time_ms and tok_tuned_ms) else "N/A"
        speedup_vs_def = f"{tok_default_ms / tok_tuned_ms:.2f}x" if (tok_default_ms and tok_tuned_ms) else "N/A"
        print(f"  Standard (einsum):     {std_time_ms:.2f} ms" if std_time_ms else "  Standard: OOM/Fail")
        print(f"  Tokamax (Untuned Def): {tok_default_ms:.2f} ms" if tok_default_ms else "  Tokamax (Def): Fail")
        print(f"  Tokamax (Tuned Splash):{tok_tuned_ms:.2f} ms" if tok_tuned_ms else "  Tokamax (Tuned): Fail")
        print(f"  Tuned Speedup vs Std:  {speedup_vs_std} | vs Untuned: {speedup_vs_def}")

        results.append({
            "seq_len": T,
            "std_ms": std_time_ms,
            "tok_def_ms": tok_default_ms,
            "tok_tuned_ms": tok_tuned_ms,
            "speedup_vs_std": speedup_vs_std,
            "speedup_vs_def": speedup_vs_def,
        })

    return results


def benchmark_full_model_step(seq_len=1024, batch_size=4, depth=12, n_embd=768, warmup=5, iters=15):
    print("\n" + "=" * 80)
    print(f"2. FULL MODEL TRAINING STEP BENCHMARK (T = {seq_len}, Batch = {batch_size})")
    print(f"Config: Depth={depth}, Emb={n_embd}, SeqLen={seq_len}, Batch={batch_size}")
    print("=" * 80)

    num_heads = n_embd // 64
    tokens_per_step = batch_size * seq_len

    variants = [
        ("Standard (einsum)", "standard", "auto"),
        ("Tokamax (Untuned Def)", "tokamax", "tokamax_default"),
        ("Tokamax (Tuned Splash)", "tokamax", "auto"),
    ]
    perf = {}

    for label, kernel, tune_mode in variants:
        print(f"\nBenchmarking full model with {label} ...")
        config = GPTConfig(
            sequence_len=seq_len,
            vocab_size=50304,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=n_embd,
            attention_kernel=kernel,
            tokamax_tune_mode=tune_mode,
        )

        model = GPT(config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4), wrt=nnx.Param)

        @nnx.jit
        def train_step(m, opt, x, y):
            def loss_fn(m):
                return m(x, targets=y)
            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            return loss

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        x_data = jax.random.randint(k1, (batch_size, seq_len), 0, 50304)
        y_data = jax.random.randint(k2, (batch_size, seq_len), 0, 50304)

        try:
            # Warmup (compilation + first iterations)
            for _ in range(warmup):
                loss = train_step(model, optimizer, x_data, y_data)
                jax.block_until_ready(loss)

            t0 = time.perf_counter()
            for _ in range(iters):
                loss = train_step(model, optimizer, x_data, y_data)
                jax.block_until_ready(loss)
            t1 = time.perf_counter()

            step_ms = (t1 - t0) / iters * 1000.0
            tok_per_sec = int(tokens_per_step / ((t1 - t0) / iters))
            perf[label] = {"step_ms": step_ms, "tok_per_sec": tok_per_sec}
            print(f"  Step Latency: {step_ms:.2f} ms | Throughput: {tok_per_sec:,} tokens/sec")
        except Exception as e:
            print(f"  Failed for {label}: {e}")
            perf[label] = None

    return perf


def main():
    iso_results = benchmark_isolated_attention(seq_lengths=(1024, 2048, 4096, 8192))
    model_results_1024 = benchmark_full_model_step(seq_len=1024, batch_size=4)
    model_results_2048 = benchmark_full_model_step(seq_len=2048, batch_size=2)

    print("\n" + "=" * 90)
    print("FINAL PERFORMANCE SUMMARY: ISOLATED ATTENTION (Forward + Backward)")
    print("=" * 90)
    print(f"{'Seq Len':<8} | {'Standard (ms)':<14} | {'Untuned Tok (ms)':<18} | {'Tuned Tok (ms)':<16} | {'Speedup vs Std':<15}")
    print("-" * 80)
    for r in iso_results:
        std_str = f"{r['std_ms']:.2f}" if r['std_ms'] else "OOM/Fail"
        def_str = f"{r['tok_def_ms']:.2f}" if r['tok_def_ms'] else "Fail"
        tun_str = f"{r['tok_tuned_ms']:.2f}" if r['tok_tuned_ms'] else "Fail"
        print(f"{r['seq_len']:<8} | {std_str:<14} | {def_str:<18} | {tun_str:<16} | {r['speedup_vs_std']:<15}")

    print("\n" + "=" * 90)
    print("FINAL PERFORMANCE SUMMARY: FULL MODEL TRAINING STEP")
    print("=" * 90)
    for name, res in [("T=1024 (Batch 4)", model_results_1024), ("T=2048 (Batch 2)", model_results_2048)]:
        print(f"\n{name}:")
        for k, v in res.items():
            if v:
                print(f"  {k:<24}: {v['step_ms']:.2f} ms ({v['tok_per_sec']:,} tokens/sec)")
            else:
                print(f"  {k:<24}: Failed")


if __name__ == "__main__":
    main()
