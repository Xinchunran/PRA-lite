#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
DEFAULT_PRAGMA_PYTHON="${HOME}/.conda/envs/pragma-lite/bin/python"
DEFAULT_PRAGMA_TORCHRUN="${HOME}/.conda/envs/pragma-lite/bin/torchrun"
if [[ -x "${DEFAULT_PRAGMA_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PRAGMA_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi
if [[ -x "${DEFAULT_PRAGMA_TORCHRUN}" ]]; then
  TORCHRUN_BIN="${TORCHRUN_BIN:-${DEFAULT_PRAGMA_TORCHRUN}}"
else
  TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Cannot resolve python interpreter. Set PYTHON_BIN explicitly." >&2
  exit 1
fi
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "Cannot resolve torchrun binary. Set TORCHRUN_BIN explicitly." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/train/pretrain_mlm.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
DATA_DIR="${DATA_DIR:-data/tokenized/transxion}"
SPLIT_DIR="${SPLIT_DIR:-data/splits/transxion}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_ddp}"
TOKENIZER_DIR="${TOKENIZER_DIR:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
RESUME_FROM="${RESUME_FROM:-}"
PROCESSED_DIR="${PROCESSED_DIR:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BACKEND="${BACKEND:-nccl}"
CHECK_SPLITS="${CHECK_SPLITS:-0}"
if [[ -z "${PRAGMA_DEBUG_ENV_FILE:-}" && -f "${PROJECT_ROOT}/.dbg/ddp-step1-hang.env" ]]; then
  export PRAGMA_DEBUG_ENV_FILE="${PROJECT_ROOT}/.dbg/ddp-step1-hang.env"
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${CHECK_SPLITS}" == "1" ]]; then
  if [[ -z "${PROCESSED_DIR}" ]]; then
    echo "CHECK_SPLITS=1 时必须提供 PROCESSED_DIR" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m src.splitter.check_splits \
    --processed_dir "${PROCESSED_DIR}" \
    --split_dir "${SPLIT_DIR}"
fi

CMD=(
  "${TORCHRUN_BIN}"
  --standalone
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  -m
  src.training.pretrain_mlm
  --config "${CONFIG}"
  --model_config "${MODEL_CONFIG}"
  --data_dir "${DATA_DIR}"
  --split_dir "${SPLIT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --backend "${BACKEND}"
)

if [[ -n "${TOKENIZER_DIR}" ]]; then
  CMD+=(--tokenizer_dir "${TOKENIZER_DIR}")
fi

if [[ -n "${MANIFEST_PATH}" ]]; then
  CMD+=(--manifest_path "${MANIFEST_PATH}")
fi

if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume_from "${RESUME_FROM}")
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "Using TORCHRUN_BIN=${TORCHRUN_BIN}"
echo "Launching DDP pretraining with ${NPROC_PER_NODE} process(es)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
