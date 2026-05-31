#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DEFAULT_PRAGMA_PYTHON="${HOME}/.conda/envs/pragma-lite/bin/python"
if [[ -x "${DEFAULT_PRAGMA_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PRAGMA_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_c}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifest.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_ROOT}/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_ibm_aml_li_medium_pragma_c}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train.log}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}/plots}"
AUTO_RESUME="${AUTO_RESUME:-0}"
CONFIG="${CONFIG:-configs/train/pretrain_mlm_pragma_c.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRECISION="${PRECISION:-bf16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_EVENTS="${MAX_EVENTS:-256}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
SPLIT_MODE="${SPLIT_MODE:-pragma_c}"

if [[ "${SPLIT_MODE}" == "random" ]]; then
  WORK_ROOT="${WORK_ROOT_RANDOM:-data/streaming/ibm_aml_li_medium}"
  MANIFEST_PATH="${MANIFEST_PATH_RANDOM:-${WORK_ROOT}/manifest.json}"
  TOKENIZER_DIR="${TOKENIZER_DIR_RANDOM:-${WORK_ROOT}/tokenizer}"
  OUTPUT_DIR="${OUTPUT_DIR_RANDOM:-runs/pretrain_ibm_aml_medium_streaming_random}"
  TRAIN_LOG="${TRAIN_LOG_RANDOM:-${OUTPUT_DIR}/train.log}"
  PLOTS_DIR="${PLOTS_DIR_RANDOM:-${OUTPUT_DIR}/plots}"
  CONFIG="${CONFIG_RANDOM:-configs/train/pretrain_mlm_small.yaml}"
fi

mkdir -p "${OUTPUT_DIR}" "${PLOTS_DIR}"

export CONFIG MODEL_CONFIG MANIFEST_PATH OUTPUT_DIR TOKENIZER_DIR PLOTS_DIR
export NPROC_PER_NODE PRECISION TRAIN_BATCH_SIZE
export DATALOADER_NUM_WORKERS DATALOADER_PREFETCH_FACTOR DATALOADER_PERSISTENT_WORKERS DATALOADER_PIN_MEMORY
export DATA_SPLIT_MODE="${DATA_SPLIT_MODE:-${SPLIT_MODE}}"
export TRAIN_SPLIT_NAME="${TRAIN_SPLIT_NAME:-train}"
export VALID_SPLIT_NAME="${VALID_SPLIT_NAME:-valid}"
export DATA_DIR="${WORK_ROOT}/tokenized_shards/shard_00000"
export SPLIT_DIR="${WORK_ROOT}/split_stub"

if [[ "${AUTO_RESUME}" == "1" && -f "${OUTPUT_DIR}/last.ckpt" ]]; then
  export RESUME_FROM="${OUTPUT_DIR}/last.ckpt"
fi

echo "[pragma_c_pretrain] PYTHON_BIN=${PYTHON_BIN}"
echo "[pragma_c_pretrain] SPLIT_MODE=${SPLIT_MODE}"
echo "[pragma_c_pretrain] WORK_ROOT=${WORK_ROOT}"
echo "[pragma_c_pretrain] MAX_EVENTS=${MAX_EVENTS}"
echo "[pragma_c_pretrain] TRAIN_LOG=${TRAIN_LOG}"

set +e
CHECK_SPLITS=0 bash scripts/train_pretrain_ddp.sh 2>&1 | tee -a "${TRAIN_LOG}"
TRAIN_EXIT_CODE=$?
set -e

if [[ -f "${OUTPUT_DIR}/metrics.jsonl" ]]; then
  "${PYTHON_BIN}" tools/plot_pretrain_metrics.py \
    --metrics_file "${OUTPUT_DIR}/metrics.jsonl" \
    --output_dir "${PLOTS_DIR}" \
    --title_prefix "IBM AML Medium PRAGMA C" || true
fi

exit "${TRAIN_EXIT_CODE}"
