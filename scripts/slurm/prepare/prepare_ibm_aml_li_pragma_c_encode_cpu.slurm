#!/bin/bash
#SBATCH --job-name=ibmaml-pragma-c-encode
#SBATCH --partition=batch
#SBATCH --account=yang_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%8
#SBATCH --output=/home/ranx/PRA-lite/logs/%x-%A_%a.out
#SBATCH --error=/home/ranx/PRA-lite/logs/%x-%A_%a.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || printf '%s' "${SLURM_SUBMIT_DIR:-/home/ranx/PRA-lite}")"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

set +u
source ~/.bashrc
set -u
conda activate pragma-lite

export WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_c}"
export NUM_SHARDS="${NUM_SHARDS:-128}"
export MAX_EVENTS="${MAX_EVENTS:-256}"

bash "${PROJECT_ROOT}/scripts/prepare/pragma_c/prepare_ibm_aml_li_pragma_c_encode_shard.sh" "${SLURM_ARRAY_TASK_ID}"
