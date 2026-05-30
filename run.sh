#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
RUN_ROOT="${PROJECT_ROOT}"
LOG_DIR="${RUN_ROOT}/logs"
RUNS_DIR="${RUN_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

cd "${PROJECT_ROOT}"

# Clean possible environment pollution from cluster shells or previous sessions.
module --force purge 2>/dev/null || true

unset PIP_INDEX_URL || true
unset PIP_EXTRA_INDEX_URL || true
unset PIP_FIND_LINKS || true
unset PIP_CACHE_DIR || true
unset PYTHONPATH || true
unset LD_PRELOAD || true

CONDA_ENV="${CONDA_ENV:-pragma-lite}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Cannot find conda initialization script. Set CONDA_SH or ensure conda is on PATH." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"

DEFAULT_TARGET_GPUS=8
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="${CUDA_VISIBLE_DEVICES}"
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
  AVAILABLE_GPU_COUNT="${#GPU_ARRAY[@]}"
else
  AVAILABLE_GPU_COUNT="$(python - <<'PY'
import subprocess

try:
    output = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    output = ""
print(sum(1 for line in output.splitlines() if line.strip()))
PY
)"
  if [[ "${AVAILABLE_GPU_COUNT}" -le 0 ]]; then
    echo "No GPUs are visible. Check CUDA/driver environment first." >&2
    exit 1
  fi
  TARGET_GPU_COUNT="${NPROC_PER_NODE:-${DEFAULT_TARGET_GPUS}}"
  if (( AVAILABLE_GPU_COUNT < TARGET_GPU_COUNT )); then
    echo "Requested ${TARGET_GPU_COUNT} GPU(s), but only ${AVAILABLE_GPU_COUNT} visible on this node." >&2
    exit 1
  fi
  GPU_LIST="$(seq -s, 0 $(( TARGET_GPU_COUNT - 1 )))"
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
fi

AVAILABLE_GPU_COUNT="${#GPU_ARRAY[@]}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${AVAILABLE_GPU_COUNT}}"
if (( AVAILABLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "CUDA_VISIBLE_DEVICES exposes ${AVAILABLE_GPU_COUNT} GPU(s), but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

TOTAL_CPU="$(nproc)"
CPU_PER_GPU=$(( TOTAL_CPU / NPROC_PER_NODE ))
if (( CPU_PER_GPU < 1 )); then
  CPU_PER_GPU=1
fi

export PYTHONPATH="${PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_PER_GPU}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_PER_GPU}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_PER_GPU}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_PER_GPU}}"
export TOKENIZERS_PARALLELISM=false
export TOKENIZE_NUM_WORKERS="${TOKENIZE_NUM_WORKERS:-${TOTAL_CPU}}"
export PRAGMA_DEBUG_ENV_FILE="${PRAGMA_DEBUG_ENV_FILE:-${PROJECT_ROOT}/.dbg/pretrain-slow.env}"
export PRAGMA_DEBUG_RUN_ID="${PRAGMA_DEBUG_RUN_ID:-post-fix}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TOTAL_CPU=${TOTAL_CPU}"
echo "CPU_PER_GPU=${CPU_PER_GPU}"

DATASET="${DATASET:-transxion_full}"
PRECISION="${PRECISION:-bf16}"
export PRECISION

RAW_PUBLIC_DIR="${RAW_PUBLIC_DIR:-data/raw/transxion_public}"
RAW_CANONICAL_DIR="${RAW_CANONICAL_DIR:-data/raw/transxion_public_canonical}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/${DATASET}.yaml}"
CONFIG="${CONFIG:-configs/train/pretrain_mlm_small.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
DO_TRAIN="${DO_TRAIN:-1}"

PROCESSED_DIR="${PROCESSED_DIR:-data/processed/${DATASET}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROCESSED_DIR}/tokenizer}"
DATA_DIR="${DATA_DIR:-${PROCESSED_DIR}/tokenized}"
SPLIT_DIR="${SPLIT_DIR:-data/splits/${DATASET}}"
TOKENIZED_BACKEND="${TOKENIZED_BACKEND:-lmdb}"
FORCE_REBUILD_TOKENIZER="${FORCE_REBUILD_TOKENIZER:-0}"
FORCE_REENCODE="${FORCE_REENCODE:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_DIR}/pretrain_ddp_${NPROC_PER_NODE}gpu_full_${RUN_ID}}"
PREP_LOG="${OUTPUT_DIR}/prepare.log"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"

{
  echo "[$(date)] Job started"
  echo "[$(date)] HOSTNAME=$(hostname)"
  echo "[$(date)] USER=$(whoami)"
  echo "[$(date)] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[$(date)] RUN_ROOT=${RUN_ROOT}"
  echo "[$(date)] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[$(date)] PWD=$(pwd)"
  echo "[$(date)] CONDA_ENV=${CONDA_ENV}"
  echo "[$(date)] CONDA_PREFIX=${CONDA_PREFIX:-unset}"
  echo "[$(date)] PYTHON=$(which python)"
  echo "[$(date)] TORCHRUN=$(which torchrun || true)"
  echo "[$(date)] NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "[$(date)] PRECISION=${PRECISION}"
  echo "[$(date)] DO_TRAIN=${DO_TRAIN}"
  echo "[$(date)] TOTAL_CPU=${TOTAL_CPU}"
  echo "[$(date)] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "[$(date)] TOKENIZE_NUM_WORKERS=${TOKENIZE_NUM_WORKERS}"
  echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[$(date)] Module list after purge:"
  module list 2>&1 || true
  echo "[$(date)] Directory permissions:"
  ls -ld "${PROJECT_ROOT}" "${RUN_ROOT}" "${LOG_DIR}" "${RUNS_DIR}" "${OUTPUT_DIR}"
  echo "[$(date)] GPU status:"
  nvidia-smi || true
  echo "[$(date)] Python/Torch check:"
  python - <<'PY'
import os
import sys

print("python:", sys.executable)
print("prefix:", sys.prefix)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
try:
    import torch

    print("torch:", torch.__version__)
    print("torch file:", torch.__file__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("TORCH_IMPORT_ERROR:", repr(exc))
    raise
PY
} 2>&1 | tee "${PREP_LOG}"

{
  echo "[$(date)] Preparing canonical raw TransXion public data"
  echo "[$(date)] RAW_PUBLIC_DIR=${RAW_PUBLIC_DIR}"
  echo "[$(date)] RAW_CANONICAL_DIR=${RAW_CANONICAL_DIR}"
} 2>&1 | tee -a "${PREP_LOG}"

python tools/prepare_transxion_public_raw.py \
  --raw_dir "${RAW_PUBLIC_DIR}" \
  --out_dir "${RAW_CANONICAL_DIR}" 2>&1 | tee -a "${PREP_LOG}"

{
  echo "[$(date)] Rebuilding events from ${DATA_CONFIG}"
} 2>&1 | tee -a "${PREP_LOG}"

python -m src.data_downloader.build_events \
  --config "${DATA_CONFIG}" 2>&1 | tee -a "${PREP_LOG}"

{
  echo "[$(date)] Building entity-level splits"
  echo "[$(date)] SPLIT_DIR=${SPLIT_DIR}"
} 2>&1 | tee -a "${PREP_LOG}"

python tools/make_entity_splits.py \
  --labels "${PROCESSED_DIR}/labels.parquet" \
  --out_dir "${SPLIT_DIR}" 2>&1 | tee -a "${PREP_LOG}"

{
  echo "[$(date)] Building tokenizer vocab"
  echo "[$(date)] PROCESSED_DIR=${PROCESSED_DIR}"
  echo "[$(date)] TOKENIZER_DIR=${TOKENIZER_DIR}"
  echo "[$(date)] FORCE_REBUILD_TOKENIZER=${FORCE_REBUILD_TOKENIZER}"
} 2>&1 | tee -a "${PREP_LOG}"

if [[ "${FORCE_REBUILD_TOKENIZER}" == "1" || ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
  python -m src.tokenizer.build_vocab \
    --processed_dir "${PROCESSED_DIR}" \
    --output_dir "${TOKENIZER_DIR}" 2>&1 | tee -a "${PREP_LOG}"
else
  echo "[$(date)] Reusing existing tokenizer at ${TOKENIZER_DIR}" 2>&1 | tee -a "${PREP_LOG}"
fi

{
  echo "[$(date)] Encoding structured PRAGMA dataset"
  echo "[$(date)] DATA_DIR=${DATA_DIR}"
  echo "[$(date)] TOKENIZED_BACKEND=${TOKENIZED_BACKEND}"
  echo "[$(date)] FORCE_REENCODE=${FORCE_REENCODE}"
} 2>&1 | tee -a "${PREP_LOG}"

HAVE_PARQUET=0
HAVE_LMDB=0
[[ -f "${DATA_DIR}/dataset.parquet" ]] && HAVE_PARQUET=1
[[ -f "${DATA_DIR}/dataset.lmdb/data.mdb" && -f "${DATA_DIR}/dataset.lmdb/lock.mdb" ]] && HAVE_LMDB=1

NEED_ENCODE=0
case "${TOKENIZED_BACKEND}" in
  parquet)
    [[ "${HAVE_PARQUET}" == "1" ]] || NEED_ENCODE=1
    ;;
  lmdb)
    [[ "${HAVE_LMDB}" == "1" ]] || NEED_ENCODE=1
    ;;
  both)
    [[ "${HAVE_PARQUET}" == "1" && "${HAVE_LMDB}" == "1" ]] || NEED_ENCODE=1
    ;;
  *)
    echo "Unsupported TOKENIZED_BACKEND=${TOKENIZED_BACKEND}; expected parquet|lmdb|both" >&2
    exit 1
    ;;
esac

if [[ "${FORCE_REENCODE}" == "1" || "${NEED_ENCODE}" == "1" ]]; then
  rm -rf "${DATA_DIR}"
  mkdir -p "${DATA_DIR}"
  python -m src.tokenizer.encode_dataset \
    --processed_dir "${PROCESSED_DIR}" \
    --tokenizer_dir "${TOKENIZER_DIR}" \
    --split_dir "${SPLIT_DIR}" \
    --output_dir "${DATA_DIR}" \
    --backend "${TOKENIZED_BACKEND}" \
    --max_events 512 \
    --max_event_tokens 24 \
    --max_profile_tokens 200 \
    --num_workers "${TOKENIZE_NUM_WORKERS}" 2>&1 | tee -a "${PREP_LOG}"
else
  echo "[$(date)] Reusing existing tokenized dataset at ${DATA_DIR}" 2>&1 | tee -a "${PREP_LOG}"
fi

python - <<PY 2>&1 | tee -a "${PREP_LOG}"
from pathlib import Path
import pandas as pd
import numpy as np

dataset_path = Path("${DATA_DIR}/dataset.parquet")
lmdb_path = Path("${DATA_DIR}/dataset.lmdb")
backend = "${TOKENIZED_BACKEND}"

print("Checking parquet dataset:", dataset_path.resolve())
print("parquet exists:", dataset_path.exists())
print("Checking lmdb dataset:", lmdb_path.resolve())
print("lmdb exists:", lmdb_path.exists())
print("backend:", backend)

required = {
    "profile_key_ids",
    "profile_value_ids",
    "profile_value_pos",
    "profile_time",
    "profile_mask",
    "event_key_ids",
    "event_value_ids",
    "event_value_pos",
    "event_token_mask",
    "event_time",
    "calendar_features",
    "event_mask",
}

if backend in {"parquet", "both"}:
    if not dataset_path.exists():
        raise SystemExit(f"Missing parquet tokenized dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    missing = sorted(required.difference(df.columns))
    print(f"dataset rows={len(df)}")
    print(f"dataset columns={list(df.columns)}")
    if missing:
        raise SystemExit(f"Missing structured columns: {missing}")

if backend in {"lmdb", "both"}:
    entity_ids_path = lmdb_path / "entity_ids.npy"
    length_path = lmdb_path / "length.txt"
    if not lmdb_path.exists():
        raise SystemExit(f"Missing lmdb tokenized dataset: {lmdb_path}")
    if not entity_ids_path.exists():
        raise SystemExit(f"Missing lmdb entity id index: {entity_ids_path}")
    if not length_path.exists():
        raise SystemExit(f"Missing lmdb length file: {length_path}")
    entity_ids = np.load(entity_ids_path)
    print(f"lmdb rows={len(entity_ids)}")

print("Structured dataset check passed")
PY

if [[ "${DO_TRAIN}" == "1" ]]; then
  if [[ "${DATASET}" == *full* && ! -d "${DATA_DIR}/dataset.lmdb" ]]; then
    echo "[$(date)] ERROR: Full-scale pretraining requires LMDB backend at ${DATA_DIR}/dataset.lmdb" >&2
    exit 1
  fi
  {
    echo "[$(date)] Starting ${NPROC_PER_NODE}-GPU DDP pretraining"
    echo "[$(date)] CONFIG=${CONFIG}"
    echo "[$(date)] MODEL_CONFIG=${MODEL_CONFIG}"
    echo "[$(date)] PRECISION=${PRECISION}"
    echo "[$(date)] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-unset}"
    echo "[$(date)] DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-unset}"
    echo "[$(date)] DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-unset}"
    echo "[$(date)] ENABLE_TF32=${ENABLE_TF32:-unset}"
    echo "[$(date)] DATA_DIR=${DATA_DIR}"
    echo "[$(date)] LMDB_DIR=${DATA_DIR}/dataset.lmdb"
    echo "[$(date)] SPLIT_DIR=${SPLIT_DIR}"
    echo "[$(date)] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[$(date)] TRAIN_LOG=${TRAIN_LOG}"
  } 2>&1 | tee "${TRAIN_LOG}"

  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  PRECISION="${PRECISION}" \
  CONFIG="${CONFIG}" \
  MODEL_CONFIG="${MODEL_CONFIG}" \
  DATA_DIR="${DATA_DIR}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  TOKENIZER_DIR="${TOKENIZER_DIR}" \
  PROCESSED_DIR="${PROCESSED_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CHECK_SPLITS=1 \
  bash scripts/train_pretrain_ddp.sh 2>&1 | tee -a "${TRAIN_LOG}"
else
  echo "[$(date)] DO_TRAIN=0, prepare only; skipping pretraining launch" 2>&1 | tee -a "${PREP_LOG}"
fi

echo "[$(date)] Job finished successfully" 2>&1 | tee -a "${TRAIN_LOG}"
