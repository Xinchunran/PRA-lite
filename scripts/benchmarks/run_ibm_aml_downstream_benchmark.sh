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

CHECKPOINT="${CHECKPOINT:-runs/pretrain_ibm_aml_li_medium_pragma_lite_full_20k_latest/best.ckpt}"
STREAM_ROOT="${STREAM_ROOT:-data/streaming/ibm_aml_li_medium_pragma_lite_full}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ibm_aml_downstream_balanced_from_best_20k}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
REPR_TYPE="${REPR_TYPE:-concat}"
CV_FOLDS="${CV_FOLDS:-3}"
MAX_HISTORY_EVENTS="${MAX_HISTORY_EVENTS:-6500}"
POSITIVE_FRACTION="${POSITIVE_FRACTION:-0.50}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${OUTPUT_DIR}"

python - <<'PY'
import importlib.util
import subprocess
import sys

missing = [pkg for pkg in ("xgboost", "catboost") if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
PY

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
  --max_history_events "${MAX_HISTORY_EVENTS}" \
  --positive_fraction "${POSITIVE_FRACTION}"
