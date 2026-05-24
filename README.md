# nanoChat.jax

A minimal, highly elegant, research-friendly JAX / Flax NNX implementation of nanoChat. Designed for clean scaling laws analysis, high-speed iteration, and robust distributed parallelism on TPUs and GPUs.

## Architectural Innovations & Production Alignment

- **Flax NNX Framework**: Pure, intuitive object-oriented state management combined with JAX's powerful functional transformations.
- **Deep Execution Profiling**: Built-in support for live JAX Profiler servers and step-based XLA trace recordings across pretraining, fine-tuning, and inference routines.
- **Decoupled Data Pipeline**: Offline pre-tokenization scripts writing highly compressed binary `.bin` token shards or clean SFT JSONLines files directly to arbitrary local directories or remote cloud storage buckets, eliminating runtime processing overhead.
- **Checkpoint Synchronization**: Prevents disk space exhaustion and network clogs by fetching only targeted single-step checkpoint directories (`--load_step <N>`) from GCS rather than downloading massive historical parameter checkpoints.

- **Universal Parameter Binding**: Automatically binds restored checkpoint weights into active optimization modules, supporting resilient shape expansions for dynamic vocabulary additions.

---

## Setup & Installation

Create a virtual environment and install dependencies (including `jax[tpu]` and `datasets`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install datasets
```

### Configuring Storage Locations

You can store dataset shards, SFT conversations, and model checkpoints in any local directory or cloud storage bucket. Define your desired base storage location by exporting the `NANOCHAT_STORAGE_ROOT` environment variable:

```bash
# Example 1: Using a direct cloud storage bucket path
export NANOCHAT_STORAGE_ROOT="gs://my-cloud-bucket/nano-chat-jax"

# Example 2: Using a local filesystem directory or mounted drive
export NANOCHAT_STORAGE_ROOT="/mnt/local_shared_disk/nano-chat-jax"
```

---

## Complete Execution Lifecycle

### 1. Dataset Ingestion & Pre-Tokenization

Download raw text parquet files or conversational turn dialogues from Hugging Face and prepare highly optimized datasets:

```bash
# Explicitly download raw pretraining parquet files using parallel worker threads
python -m nanochat.dataset --num-files 5 --num-workers 4

# Pre-tokenize pretraining dataset offline and persist binary shards directly to storage root
python scripts/prepare_dataset_gcs.py \
    --target_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --doc_limit 10000

# Compile medium-sized high-quality SFT dialogue dataset from Ultrachat and upload structured JSONLines directly to GCS
python scripts/prepare_sft_gcs.py \
    --gcs_path "$NANOCHAT_STORAGE_ROOT/sft_dataset/sft_conversations.jsonl" \
    --limit 25000
```

*Note*: You can customize the base download directory for raw parquet files by setting the `NANOCHAT_BASE_DIR` environment variable.

### 2. Base Model Pretraining (Grain Tuners & MaxText Checkpoint Architecture)

Train a base model from scratch using Grain shared-memory streaming over pre-tokenized binary shards. Features robust MaxText cloud alignment, step resumption from cloud checkpoints, and animated progress spinners:

```bash
python scripts/base_train.py \
    --depth 12 \
    --num_iterations 35000 \
    --device_batch_size 4 \
    --dp 1 --fsdp 8 --tp 1 \
    --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --grain_workers 16 \
    --grain_buffer_size 512 \
    --ckpt_every 500 \
    --model_tag base_trained_gpt2_full
```

### 3. Supervised Fine-Tuning (SFT) with Targeted Single-Step Syncing

Fine-tune your model on conversational turns using target mask filtering (training on assistant responses while ignoring prompts). Bypass corrupted preemption saves by explicitly specifying healthy pretraining steps (`--load_step`):

```bash
python scripts/chat_sft.py \
    --sft_dataset_path "$NANOCHAT_STORAGE_ROOT/sft_dataset/sft_conversations.jsonl" \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --load_model_tag base_trained_gpt2_full \
    --save_model_tag final_sft_model \
    --load_step 30000 \
    --num_iterations 500 \
    --device_batch_size 4 \
    --max_seq_len 1024 \
    --dp 1 --fsdp 8 --tp 1 \
    --ckpt_every 100
```

### 4. Interactive CLI Generation (Dual-Mode Evaluation)

Evaluate autoregressive text streaming on either conversational assistant models or raw base models using `--raw_pretrain` to skip special delimiters:

```bash
# Mode 1: Evaluate conversational fine-tuning response alignment
python scripts/chat_cli.py \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --load_model_tag final_sft_model \
    --prompt "What is the capital of France?" \
    --temperature 0.7

# Mode 2: Evaluate pure base pretraining knowledge distributions without special prompt delimiters
python scripts/chat_cli.py \
    --gcs_bucket "$NANOCHAT_STORAGE_ROOT/checkpoints" \
    --load_model_tag base_trained_gpt2_full \
    --load_step 30000 \
    --prompt "The capital of France is" \
    --raw_pretrain \
    --temperature 0.8
```

---

## Profiling Server & Detailed Tracing

All three core execution scripts (`base_train.py`, `chat_sft.py`, `chat_cli.py`) natively integrate JAX profiling functionality, allowing you to examine XLA compile times, communication stalls, and operator performance across TensorBoard or Perfetto:

```bash
# Option 1: Start a background JAX profiler server on a specific port for live capture
python scripts/base_train.py --shard_dir "$NANOCHAT_STORAGE_ROOT/dataset_tokens" --profile_server_port 9999

# Option 2: Automatically record and dump a step-based XLA trace between specific iterations
python scripts/chat_sft.py --load_model_tag base_trained_gpt2_full --profile_start 5 --profile_end 15 --profile_dir "/tmp/my_traces"

# Option 3: Trace generation latency per token during autoregressive sampling
python scripts/chat_cli.py --load_model_tag final_sft_model --profile_start 1 --profile_end 5
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
