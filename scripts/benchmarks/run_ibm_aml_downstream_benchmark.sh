#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || { cd "${SCRIPT_DIR}/../.." && pwd; })"
cd "${PROJECT_ROOT}"

set +u
source ~/.bashrc
set -u
conda activate pragma-lite
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

CHECKPOINT="${CHECKPOINT:-runs/pretrain_ibm_aml_li_medium_pragma_lite_full_20k_latest/last.ckpt}"
STREAM_ROOT="${STREAM_ROOT:-data/streaming/ibm_aml_li_medium_pragma_lite_full}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ibm_aml_downstream_from_20k_latest}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
REPR_TYPE="${REPR_TYPE:-concat}"
CV_FOLDS="${CV_FOLDS:-3}"
MAX_HISTORY_EVENTS="${MAX_HISTORY_EVENTS:-6500}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${OUTPUT_DIR}"

python scripts/benchmarks/run_ibm_aml_downstream_benchmark.py \
  --checkpoint "${CHECKPOINT}" \
  --stream_root "${STREAM_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_size "${SAMPLE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --repr_type "${REPR_TYPE}" \
  --cv_folds "${CV_FOLDS}" \
  --max_history_events "${MAX_HISTORY_EVENTS}"
