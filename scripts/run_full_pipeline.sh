#!/bin/bash
set -e

WORKSPACE_DIR="/mnt/v7x8-disk/iharsh_workspace"
CODE_DIR="${WORKSPACE_DIR}/code/nanochat.jax"
VENV_PYTHON="${WORKSPACE_DIR}/venv/bin/python"
RESULTS_DIR="${WORKSPACE_DIR}/results"
LOG_FILE="${RESULTS_DIR}/pipeline.log"

mkdir -p "${RESULTS_DIR}"
echo "=== Starting Full 3B Token Pretraining & SFT Pipeline ===" | tee "${LOG_FILE}"
date | tee -a "${LOG_FILE}"

# Step 1: Pretraining (Prepare 3B tokens + Train on 8 devices with Tokamax)
echo -e "\n--- Step 1/3: Nemotron Pretraining (3B Tokens) ---" | tee -a "${LOG_FILE}"
cd "${CODE_DIR}"
${VENV_PYTHON} scripts/run_nemotron_pretrain.py \
    --data_dir="${WORKSPACE_DIR}/data/nemotron_pretrain" \
    --target_tokens=3000000000 \
    --shard_size=10000000 \
    --depth=12 \
    --max_seq_len=1024 \
    --num_iterations=5722 \
    --device_batch_size=8 \
    --total_batch_size=524288 \
    --learning_rate=6e-4 \
    --weight_decay=0.1 \
    --dp=8 \
    --fsdp=1 \
    --tp=1 \
    --attention_kernel=tokamax \
    --tokamax_tune_mode=auto \
    --checkpoint_dir="${WORKSPACE_DIR}/checkpoints" \
    --model_tag=nemotron-base-3b \
    --ckpt_every=500 \
    --eval_every=250 \
    --metrics_json_path="${RESULTS_DIR}/pretrain_metrics.json" 2>&1 | tee -a "${LOG_FILE}"

echo -e "\n--- Step 1 Complete: Base Pretraining Finished ---" | tee -a "${LOG_FILE}"
date | tee -a "${LOG_FILE}"

# Step 2: SFT Fine-Tuning
echo -e "\n--- Step 2/3: Nemotron Supervised Fine-Tuning (SFT) ---" | tee -a "${LOG_FILE}"
${VENV_PYTHON} scripts/run_nemotron_sft.py \
    --train_only \
    --data_dir="${WORKSPACE_DIR}/data/nemotron_sft" \
    --checkpoint_dir="${WORKSPACE_DIR}/checkpoints" \
    --load_model_tag=nemotron-base-3b \
    --save_model_tag=nemotron-sft-run \
    --load_step=-1 \
    --depth=12 \
    --max_seq_len=1024 \
    --num_iterations=300 \
    --device_batch_size=2 \
    --learning_rate=5e-5 \
    --dp=8 \
    --fsdp=1 \
    --tp=1 \
    --attention_kernel=tokamax \
    --tokamax_tune_mode=auto \
    --ckpt_every=50 \
    --metrics_json_path="${RESULTS_DIR}/sft_metrics.json" 2>&1 | tee -a "${LOG_FILE}"

echo -e "\n--- Step 2 Complete: SFT Fine-Tuning Finished ---" | tee -a "${LOG_FILE}"
date | tee -a "${LOG_FILE}"

# Step 3: Plot Generation
echo -e "\n--- Step 3/3: Generating Metrics Plots ---" | tee -a "${LOG_FILE}"
cd "${RESULTS_DIR}"
${VENV_PYTHON} plot_metrics.py 2>&1 | tee -a "${LOG_FILE}"

echo "=== Pipeline Completed Successfully! ===" | tee -a "${LOG_FILE}"
date | tee -a "${LOG_FILE}"
