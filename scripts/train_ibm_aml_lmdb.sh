#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CONFIG="${CONFIG:-configs/train/pretrain_mlm_small.yaml}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
export DATA_DIR="${DATA_DIR:-data/processed/ibm_aml_full/tokenized}"
export SPLIT_DIR="${SPLIT_DIR:-data/splits/ibm_aml_full}"
export PROCESSED_DIR="${PROCESSED_DIR:-data/processed/ibm_aml_full}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-data/processed/ibm_aml_full/tokenizer}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_ibm_aml_full}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PRECISION="${PRECISION:-bf16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
export DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
export ENABLE_TF32="${ENABLE_TF32:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "${DATA_DIR}/dataset.lmdb" ]]; then
  echo "Missing LMDB dataset at ${DATA_DIR}/dataset.lmdb. Run prepare first." >&2
  exit 1
fi

CHECK_SPLITS=1 bash scripts/train_pretrain_ddp.sh
