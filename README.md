# nanoChat.jax

A minimal, highly elegant, research-friendly JAX / Flax NNX implementation of nanoChat. Designed for clean scaling laws analysis, fast iteration, and robust distributed parallelism on TPUs and GPUs.

## Architecture Highlights

- **Flax NNX Framework**: Pure, intuitive object-oriented state management combined with JAX's powerful transformations.
- **Distributed Mesh Sharding**: Out-of-the-box support for Data Parallelism (DP), Fully Sharded Data Parallelism (FSDP / ZeRO), and Tensor Parallelism (TP) via JAX Mesh and NamedSharding.
- **Decoupled Data Pipeline**: Standalone offline pre-tokenization scripts writing highly compressed binary `.bin` token shards directly to arbitrary directories or cloud buckets, eliminating runtime tokenization overhead.
- **Flawless Resilient Checkpointing**: Natively checkpoint and restore model/optimizer state through Orbax across local file systems or cloud URIs, with dynamic sequence length and vocabulary resizing resilience.
- **Deep Execution Profiling**: Embedded support for live JAX Profiler servers and step-based XLA trace recordings across pretraining, fine-tuning, and inference routines.

---

## Setup & Installation

Create a virtual environment and install dependencies (including `jax[tpu]`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuring Storage Location

You can store dataset shards and model checkpoints in any local directory or cloud storage bucket. Define your desired base storage location by exporting the `NANOCHAT_STORAGE_ROOT` environment variable:

```bash
# Example 1: Using a direct cloud storage bucket path
export NANOCHAT_STORAGE_ROOT="gs://my-cloud-bucket/nano-chat-jax"

# Example 2: Using a local filesystem directory or mounted drive
export NANOCHAT_STORAGE_ROOT="/mnt/local_shared_disk/nano-chat-jax"
```

---

## Complete Execution Lifecycle

### 1. Dataset Ingestion & Offline Pre-Tokenization

Download raw parquet files from Hugging Face and prepare highly compressed binary `.bin` token shards with visible progress metrics (`tqdm`):

```bash
# Explicitly download raw parquet files using parallel worker threads
python -m nanochat.dataset --num-files 5 --num-workers 4

# Pre-tokenize dataset offline and persist binary shards directly to your configured storage root
python scripts/prepare_dataset_gcs.py \
    --target_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --doc_limit 10000
```

*Note*: You can customize the base download directory for raw parquet files by setting the `NANOCHAT_BASE_DIR` environment variable.

### 2. Base Model Pretraining

Train a base model from scratch using Grain shared-memory streaming over pre-tokenized binary shards.

```bash
python scripts/base_train.py \
    --depth 12 \
    --num_iterations 1000 \
    --device_batch_size 4 \
    --dp 1 --fsdp 1 --tp 1 \
    --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --model_tag gpt2-base-pretraining
```

### 3. Supervised Fine-Tuning (SFT)

Tune the base model on conversation turns with target mask filtering (learning on assistant responses while ignoring prompts). Features automatic fallback vocabulary growth when special conversational tags are introduced.

```bash
python scripts/chat_sft.py \
    --load_model_tag gpt2-base-pretraining \
    --save_model_tag gpt2-chat-sft \
    --num_iterations 200 \
    --learning_rate 5e-5 \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints"
```

### 4. Interactive CLI Generation

Run autoregressive generation using KV cache streaming over restored SFT parameters.

```bash
python scripts/chat_cli.py \
    --load_model_tag gpt2-chat-sft \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --prompt "The capital of France is" \
    --temperature 0.7
```

---

## Profiling Server & Detailed Tracing

All three execution scripts (`base_train.py`, `chat_sft.py`, `chat_cli.py`) natively integrate JAX profiling functionality, allowing you to examine XLA compile times, communication stalls, and operator performance across TensorBoard or Perfetto:

```bash
# Option 1: Start a background JAX profiler server on a specific port for live capture
python scripts/base_train.py --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" --profile_server_port 9999

# Option 2: Automatically record and dump a step-based XLA trace between specific iterations
python scripts/chat_sft.py --load_model_tag gpt2-base-pretraining --profile_start 5 --profile_end 15 --profile_dir "/tmp/my_traces"

# Option 3: Trace generation latency per token during autoregressive sampling
python scripts/chat_cli.py --load_model_tag gpt2-chat-sft --profile_start 1 --profile_end 5
```

---

## Distributed Parallelism Strategy

Combine any parallelism configuration seamlessly via command line flags:
- `--dp <N>`: Replicate parameters across data batch dimension.
- `--fsdp <N>`: Fully shard parameter optimization state across batch dimension.
- `--tp <N>`: Partition Transformer Linear matrices across tensor attention heads.

## Unit Testing & Verification

Run the local pytest suite to verify structural components, loss functions, distributed sharding configurations, and KV caching across active hardware slices:

```bash
pytest tests/ -v
```
