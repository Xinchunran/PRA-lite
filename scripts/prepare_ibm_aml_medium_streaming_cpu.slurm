#!/bin/bash
#SBATCH --job-name=ibmaml-medium-cpu
#SBATCH --partition=batch
#SBATCH --account=yang_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --array=15-125%8
#SBATCH --output=/home/ranx/PRA-lite/logs/%x-%A_%a.out
#SBATCH --error=/home/ranx/PRA-lite/logs/%x-%A_%a.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/ranx/PRA-lite}"
cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/logs"

set +u
source ~/.bashrc
set -u
conda activate pragma-lite

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium}"
export RAW_SHARD_DIR="${RAW_SHARD_DIR:-${WORK_ROOT}/raw_shards}"
export PROCESSED_SHARD_ROOT="${PROCESSED_SHARD_ROOT:-${WORK_ROOT}/processed_shards}"
export TOKENIZED_SHARD_ROOT="${TOKENIZED_SHARD_ROOT:-${WORK_ROOT}/tokenized_shards}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_ROOT}/tokenizer}"
export MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifest.json}"

export MAX_EVENTS="${MAX_EVENTS:-256}"
export MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
export MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
export HASH_SPLIT_SEED="${HASH_SPLIT_SEED:-26}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
export VALID_FRAC="${VALID_FRAC:-0.1}"
export TOKENIZE_NUM_WORKERS="${TOKENIZE_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME="$(printf 'shard_%05d' "${SHARD_INDEX}")"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] shard=${SHARD_NAME}"
echo "[$(date)] cpus=${SLURM_CPUS_PER_TASK:-unknown} mem=${SLURM_MEM_PER_NODE:-unknown}"
echo "[$(date)] tokenizer=${TOKENIZER_DIR}"
echo "[$(date)] manifest=${MANIFEST_PATH}"

bash "${PROJECT_ROOT}/scripts/prepare_ibm_aml_medium_streaming_shard.sh" "${SHARD_NAME}"
