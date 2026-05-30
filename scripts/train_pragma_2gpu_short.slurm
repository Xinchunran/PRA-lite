#!/bin/bash
#SBATCH --job-name=pragma-ddp-short
#SBATCH --partition=batch_gpu
#SBATCH --account=csb_gpu_acc
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source ~/.bashrc
conda activate pragma-lite

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DATASET="transxion"
DATA_CONFIG="configs/data/${DATASET}.yaml"
CONFIG="configs/train/pretrain_mlm.yaml"
MODEL_CONFIG="configs/model/pragma_lite_small.yaml"
PROCESSED_DIR="data/processed/${DATASET}"
TOKENIZER_DIR="${PROCESSED_DIR}/tokenizer"
DATA_DIR="${PROCESSED_DIR}/tokenized"
SPLIT_DIR="data/splits/${DATASET}"
OUTPUT_DIR="runs/pretrain_ddp_2gpu_short"
PREP_LOG="${OUTPUT_DIR}/prepare.log"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

mkdir -p logs "${OUTPUT_DIR}"

{
  echo "[$(date)] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[$(date)] Rebuilding structured PRAGMA dataset"
  echo "[$(date)] Using OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  nvidia-smi || true
} 2>&1 | tee "${PREP_LOG}"

python -m src.data_downloader.build_events \
  --config "${DATA_CONFIG}" 2>&1 | tee -a "${PREP_LOG}"

python -m src.tokenizer.build_vocab \
  --processed_dir "${PROCESSED_DIR}" \
  --output_dir "${TOKENIZER_DIR}" 2>&1 | tee -a "${PREP_LOG}"

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

python -m src.tokenizer.encode_dataset \
  --processed_dir "${PROCESSED_DIR}" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --output_dir "${DATA_DIR}" \
  --max_events 512 \
  --max_event_tokens 24 \
  --max_profile_tokens 200 2>&1 | tee -a "${PREP_LOG}"

python - <<'PY' 2>&1 | tee -a "${PREP_LOG}"
import pandas as pd

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
df = pd.read_parquet("data/processed/transxion/tokenized/dataset.parquet")
missing = sorted(required.difference(df.columns))
print(f"dataset rows={len(df)}")
print(f"dataset columns={list(df.columns)}")
if missing:
    raise SystemExit(f"Missing structured columns: {missing}")
print("Structured dataset check passed")
PY

{
  echo "[$(date)] Starting 2-GPU DDP pretraining"
  echo "[$(date)] Logs: ${TRAIN_LOG}"
} 2>&1 | tee "${TRAIN_LOG}"

NPROC_PER_NODE=2 \
CONFIG="${CONFIG}" \
MODEL_CONFIG="${MODEL_CONFIG}" \
DATA_DIR="${DATA_DIR}" \
SPLIT_DIR="${SPLIT_DIR}" \
TOKENIZER_DIR="${TOKENIZER_DIR}" \
PROCESSED_DIR="${PROCESSED_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CHECK_SPLITS=1 \
bash scripts/train_pretrain_ddp.sh 2>&1 | tee -a "${TRAIN_LOG}"
