# Hardware-Accelerated Attention Kernels (Tokamax Support)

`nanoChat.jax` supports configurable attention kernels across all pretraining, fine-tuning, and inference workflows via the `--attention_kernel` command-line argument.

---

## 1. Supported Attention Kernels

- `--attention_kernel standard` (**Default**): Pure JAX einsum attention with causal masking. Compatible across CPU, GPU, and TPU architectures.
- `--attention_kernel tokamax`: Hardware-accelerated FlashAttention via **Tokamax** (`tokamax.dot_product_attention`).
  - **TPU**: Lowers to Pallas / Mosaic TPU Splash Attention kernels.
  - **GPU**: Lowers to Triton / CuDNN FlashAttention.
  - Reduces attention memory complexity from quadratic $O(T^2)$ to linear $O(T)$ and substantially increases token throughput.

---

## 2. Kernel Tuning Modes (`--tokamax_tune_mode`)

On TPUs, Splash Attention performance strongly depends on the query/key/value tile dimensions and memory layouts chosen for the hardware VPU/VMX matrix units. nanoChat.jax provides three tuning modes:

### `auto` (Default)
Applies optimal, hardware-verified configurations discovered via empirical benchmarking on TPU v7x:

| Sequence Length | Forward Tiles (`bq / bkv / bkv_compute`) | Backward Tiles (`bq_dkv / bkv_dkv / bkv_dkv_compute`) | Layout (`Q / K / V`) | Experimental Scheduler |
| :--- | :--- | :--- | :--- | :--- |
| **$T \le 1024$** | `1024 / 1024 / 512` | `1024 / 1024 / 1024` | `SEQ_MINOR` | Disabled |
| **$T \ge 2048$** | `512 / 2048 / 2048` | `1024 / 2048 / 1024` | `HEAD_DIM_MINOR` | Enabled |

### `tokamax_default`
Uses Tokamax's built-in heuristic defaults (`block_q=128, block_kv=128, block_kv_compute=128`). Useful as an un-tuned baseline.

### `custom`
Enables user-specified block dimensions and memory layouts via the granular CLI overrides detailed below.

---

## 3. Custom Tile & Layout Overrides

When using `--tokamax_tune_mode custom` (or overriding specific defaults in `auto` mode), you can configure each kernel tile parameter individually:

### Forward Pass Tile Sizes
- `--tokamax_block_q <INT>`: Forward query tile size. Must be a multiple of 128 (e.g. 128, 256, 512, 1024).
- `--tokamax_block_kv <INT>`: Forward key/value tile size (e.g. 512, 1024, 2048).
- `--tokamax_block_kv_compute <INT>`: Forward key/value compute chunk size within the tile loop (e.g. 512, 1024, 2048).

### Backward Pass Tile Sizes (VJP)
- `--tokamax_block_q_dkv <INT>`: Backward query tile size (e.g. 512, 1024).
- `--tokamax_block_kv_dkv <INT>`: Backward key/value tile size (e.g. 1024, 2048).
- `--tokamax_block_kv_dkv_compute <INT>`: Backward KV compute chunk size (e.g. 512, 1024).

### Memory Layouts
- `--tokamax_q_layout <head_dim_minor|seq_minor>`: Query memory layout.
  - `seq_minor` (default for $T \le 1024$): Higher throughput for shorter sequence lengths.
  - `head_dim_minor` (default for $T \ge 2048$): Optimal for longer sequences.
- `--tokamax_k_layout <head_dim_minor|seq_minor>`: Key memory layout.
- `--tokamax_v_layout <head_dim_minor|seq_minor>`: Value memory layout.

### Experimental Warp Scheduler
- `--tokamax_use_experimental_scheduler <True|False>`: Enables the Pallas experimental warp scheduler for overlapping communication and compute.

---

## 4. Distributed Multi-Device Execution (`q_sharding`)

Mosaic TPU custom calls cannot be automatically partitioned by JAX's GSPMD partitioner in distributed topologies. 

`nanoChat.jax` automatically inspects the active device mesh (such as `dp=1, fsdp=8, tp=1`) and derives the appropriate `q_sharding`:
```python
q_sharding = NamedSharding(mesh, PartitionSpec(("dp", "fsdp"), None, None, None))
```
This is passed directly to `tokamax.dot_product_attention`, triggering Tokamax's built-in `shard_map` wrapper so the Mosaic custom call executes locally on each TPU device without compilation errors.

---

## 5. Performance Benchmarks

Measured on **Google Cloud TPU v7x-8** (8 devices, 4 chips) across batch size 8 and hidden dimension 768 (12 heads, head dim 64):

### Isolated Attention Kernel Runtime (Forward + Backward)

| Sequence Length | Standard Einsum | Tokamax (Default Heuristics) | Tokamax (Tuned `auto`) | Speedup vs Standard | Speedup vs Untuned Tokamax |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$T = 512$** | 0.81 ms | 4.88 ms | **0.60 ms** | **1.34x faster** | **8.1x faster** |
| **$T = 1024$** | 1.83 ms | 10.74 ms | **0.65 ms** | **2.80x faster** | **16.5x faster** |
| **$T = 2048$** | 5.37 ms | 22.04 ms | **2.62 ms** | **2.05x faster** | **8.4x faster** |

### End-to-End Base Pretraining Throughput (12-layer GPT)

- **$T = 1024$**: **416,739 tokens/sec**
- **$T = 2048$**: **1.16x faster** full training step throughput compared to standard einsum.

---

## 6. CLI Examples

### Base Pretraining with Tuned Tokamax (Recommended)
```bash
python scripts/base_train.py \
    --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --attention_kernel tokamax \
    --tokamax_tune_mode auto \
    --max_seq_len 1024 \
    --device_batch_size 2 \
    --dp 1 --fsdp 8 --tp 1
```

### Base Pretraining with Custom Block Size Overrides
```bash
python scripts/base_train.py \
    --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --attention_kernel tokamax \
    --tokamax_tune_mode custom \
    --tokamax_block_q 512 \
    --tokamax_block_kv 1024 \
    --tokamax_block_kv_compute 512 \
    --tokamax_q_layout seq_minor \
    --max_seq_len 1024
```

### Chat SFT Fine-Tuning with Tokamax
```bash
python scripts/chat_sft.py \
    --sft_dataset_path "$NANOCHAT_STORAGE_ROOT/sft_dataset/sft_conversations.jsonl" \
    --attention_kernel tokamax \
    --load_model_tag nemotron-base-run \
    --save_model_tag nemotron-sft-run \
    --max_seq_len 1024 \
    --device_batch_size 1
```

### Interactive Generation with Tokamax
```bash
python scripts/chat_cli.py \
    --attention_kernel tokamax \
    --load_model_tag nemotron-sft-run \
    --prompt "Explain quantum entanglement in simple terms."
```
