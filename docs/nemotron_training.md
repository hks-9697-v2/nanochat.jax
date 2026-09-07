# NVIDIA Nemotron-3 Ultra Pretraining & SFT Fine-Tuning Pipelines

`nanoChat.jax` provides dedicated, turnkey automation scripts for streaming, formatting, and training on the **NVIDIA Nemotron-3 Ultra** dataset ([`nvidia/Nemotron-RL-Ultra-Training-Blends`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends) on Hugging Face).

The dataset combines multiple high-quality subsets spanning synthetic reasoning, complex tool usage, instruction-following benchmarks, and RLHF alignment:
- `reasoning`: Multi-step reasoning chains and mathematical solutions.
- `rlhf`: Preference-tuned pairs and safe conversational responses.
- `ifbench`: Rigorous instruction-following constraint evaluations.
- `rlvr1` / `rlvr2`: Reinforcement learning verifiable rewards.
- `swe` / `mopd`: Software engineering and multi-operation program data.

---

## Architecture of the Automation Scripts

Both automation scripts support flexible execution modes:
1. **Full End-to-End**: Automatically downloads/streams the dataset, encodes the data, and immediately launches distributed training.
2. **`--prepare_only`**: Preprocesses and persists the dataset locally or to cloud storage without starting training.
3. **`--train_only`**: Skips the dataset download step and begins training immediately on existing local shards/files.

---

## 1. Pretraining Pipeline (`scripts/run_nemotron_pretrain.py`)

This script streams text from Nemotron subsets, tokenizes documents using `tiktoken` (delimited by the GPT-2 BOS token `50256`), writes compressed binary `.bin` shards with the standard 256-int32 header (`magic=20240520, version=1`), and executes distributed pretraining via `scripts/base_train.py`.

### Key Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_name` | `str` | `nvidia/Nemotron-RL-Ultra-Training-Blends` | Hugging Face dataset identifier |
| `--configs` | `str` | `reasoning,mopd,ifbench,swe,rlhf` | Comma-separated dataset subsets to stream |
| `--data_dir` | `str` | `/mnt/v7x8-disk/iharsh_workspace/data/nemotron_pretrain` | Destination directory for `.bin` token shards |
| `--checkpoint_dir` | `str` | `/mnt/v7x8-disk/iharsh_workspace/checkpoints` | Root directory for saving checkpoints |
| `--model_tag` | `str` | `nemotron-base-run` | Subdirectory tag name for saved base checkpoints |
| `--max_docs` | `int` | `25000` | Maximum documents to stream and encode (`-1` = unlimited) |
| `--shard_size` | `int` | `1000000` | Tokens per binary shard (`.bin`) |
| `--prepare_only` | `flag` | `False` | Only download and tokenize shards; do not launch training |
| `--train_only` | `flag` | `False` | Skip dataset prep and launch training on existing shards |
| `--depth` | `int` | `12` | Transformer layer depth |
| `--max_seq_len` | `int` | `1024` | Sequence context window length |
| `--num_iterations` | `int` | `200` | Pretraining optimization iterations |
| `--device_batch_size`| `int` | `4` | Batch size per device (global batch = `device_batch_size * dp * fsdp`) |
| `--total_batch_size` | `int` | `32768` | Total effective token batch size |
| `--learning_rate` | `float` | `3e-4` | Peak learning rate for AdamW |
| `--attention_kernel`| `str` | `tokamax` | Attention kernel implementation (`tokamax` or `standard`) |
| `--tokamax_tune_mode`| `str` | `auto` | Tuning mode (`auto`, `tokamax_default`, `custom`) |
| `--dp` / `--fsdp` / `--tp` | `int` | `1 / 8 / 1` | Distributed device mesh parallelism shape |

### Pretraining Usage Examples

#### Run Full End-to-End Pretraining
```bash
python scripts/run_nemotron_pretrain.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_pretrain \
    --checkpoint_dir /mnt/v7x8-disk/iharsh_workspace/checkpoints \
    --model_tag nemotron-base-run \
    --max_docs 50000 \
    --shard_size 1000000 \
    --num_iterations 1000 \
    --device_batch_size 2 \
    --attention_kernel tokamax
```

#### Prepare Shards Only (Offline Data Prep)
```bash
python scripts/run_nemotron_pretrain.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_pretrain \
    --max_docs 100000 \
    --shard_size 2000000 \
    --prepare_only
```

#### Train Only on Existing Pretokenized Shards
```bash
python scripts/run_nemotron_pretrain.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_pretrain \
    --checkpoint_dir /mnt/v7x8-disk/iharsh_workspace/checkpoints \
    --model_tag nemotron-base-run \
    --num_iterations 5000 \
    --device_batch_size 4 \
    --train_only
```

---

## 2. Supervised Fine-Tuning Pipeline (`scripts/run_nemotron_sft.py`)

This script extracts multi-turn conversational interactions from Nemotron records, normalizes them into strictly alternating `user`/`assistant` turns (folding system instructions into user prompts and coalescing same-role messages), formats them into JSONLines (`nemotron_sft.jsonl`), restores a base model checkpoint, and executes distributed SFT fine-tuning via `scripts/chat_sft.py` with ChatML token masking.

### Key Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_name` | `str` | `nvidia/Nemotron-RL-Ultra-Training-Blends` | Hugging Face dataset identifier |
| `--configs` | `str` | `reasoning,rlhf,ifbench,rlvr1` | Subsets to stream for conversational dialogues |
| `--data_dir` | `str` | `/mnt/v7x8-disk/iharsh_workspace/data/nemotron_sft` | Directory to save `nemotron_sft.jsonl` |
| `--checkpoint_dir` | `str` | `/mnt/v7x8-disk/iharsh_workspace/checkpoints` | Root directory containing base checkpoints and saving SFT checkpoints |
| `--load_model_tag` | `str` | `nemotron-base-run` | Base model checkpoint tag to restore weights from |
| `--save_model_tag` | `str` | `nemotron-sft-run` | Tag name for saved SFT fine-tuned checkpoints |
| `--load_step` | `int` | `-1` | Specific base checkpoint step to restore (`-1` = latest) |
| `--max_dialogues` | `int` | `25000` | Maximum conversational dialogues to extract |
| `--prepare_only` | `flag` | `False` | Only extract and format `nemotron_sft.jsonl` |
| `--train_only` | `flag` | `False` | Skip extraction and launch SFT on existing JSONL |
| `--num_iterations` | `int` | `300` | SFT optimization steps |
| `--device_batch_size`| `int` | `2` | Batch size per device |
| `--learning_rate` | `float` | `5e-5` | Peak fine-tuning learning rate |
| `--attention_kernel`| `str` | `tokamax` | Attention kernel implementation (`tokamax` or `standard`) |
| `--ckpt_every` | `int` | `50` | Interval for saving interim SFT checkpoints |

### SFT Usage Examples

#### Run Full End-to-End SFT (Restoring Base Step 1000)
```bash
python scripts/run_nemotron_sft.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_sft \
    --checkpoint_dir /mnt/v7x8-disk/iharsh_workspace/checkpoints \
    --load_model_tag nemotron-base-run \
    --load_step 1000 \
    --save_model_tag nemotron-sft-run \
    --max_dialogues 25000 \
    --num_iterations 500 \
    --device_batch_size 2 \
    --learning_rate 5e-5 \
    --attention_kernel tokamax
```

#### Prepare Dialogue Dataset Only
```bash
python scripts/run_nemotron_sft.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_sft \
    --max_dialogues 50000 \
    --prepare_only
```

#### Train SFT Only on Existing JSONLines
```bash
python scripts/run_nemotron_sft.py \
    --data_dir /mnt/v7x8-disk/iharsh_workspace/data/nemotron_sft \
    --checkpoint_dir /mnt/v7x8-disk/iharsh_workspace/checkpoints \
    --load_model_tag nemotron-base-run \
    --save_model_tag nemotron-sft-run \
    --num_iterations 300 \
    --train_only
```

---

## 3. Storage & Workspace Guidelines

On Google Cloud TPU VMs, root overlay filesystems are typically constrained in size (e.g. 50-100GB). Always use the dedicated high-capacity storage volume (such as `/mnt/v7x8-disk/` or a mounted GCS FUSE / persistent disk) for all datasets, token shards, virtual environments, and Orbax checkpoints:

```bash
# Recommended directory layout:
/mnt/v7x8-disk/iharsh_workspace/
├── code/nanochat.jax/
├── data/
│   ├── nemotron_pretrain/   # Binary .bin token shards
│   └── nemotron_sft/        # nemotron_sft.jsonl
└── checkpoints/
    ├── nemotron-base-run/   # Base pretraining Orbax checkpoints
    └── nemotron-sft-run/    # Fine-tuned SFT Orbax checkpoints
```
