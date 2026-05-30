#!/bin/bash
#SBATCH --job-name=ibmaml-pragma-c-prep
#SBATCH --partition=batch
#SBATCH --account=yang_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ranx/PRA-lite/logs/%x-%A.out
#SBATCH --error=/home/ranx/PRA-lite/logs/%x-%A.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/ranx/PRA-lite}"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

set +u
source ~/.bashrc
set -u
conda activate pragma-lite

export WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_c}"
export RAW_DIR="${RAW_DIR:-data/raw/ibm_aml}"
export RAW_CSV="${RAW_CSV:-LI-Medium_Trans.csv}"

bash "${PROJECT_ROOT}/scripts/prepare_ibm_aml_li_pragma_c.sh"
