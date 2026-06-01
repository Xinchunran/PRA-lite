#!/bin/bash
#SBATCH --job-name=pra-lite-ibm-downstream
#SBATCH --partition=batch
#SBATCH --account=yang_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --output=/home/ranx/PRA-lite/logs/%x-%j.out
#SBATCH --error=/home/ranx/PRA-lite/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || printf '%s' "${SLURM_SUBMIT_DIR:-/home/ranx/PRA-lite}")"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

export OUTPUT_DIR="${OUTPUT_DIR:-runs/ibm_aml_downstream_balanced_from_best_20k}"
export POSITIVE_FRACTION="${POSITIVE_FRACTION:-0.50}"
export DEVICE="${DEVICE:-cpu}"

bash "${PROJECT_ROOT}/scripts/benchmarks/run_ibm_aml_downstream_benchmark.sh"
