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

RAW_DIR="${RAW_DIR:-data/raw/ibm_aml}"
RAW_CSV="${RAW_CSV:-LI-Medium_Trans.csv}"
WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_lite_full}"
MAX_HISTORY_EVENTS="${MAX_HISTORY_EVENTS:-6500}"
PROFILE_SAMPLE_LIMIT="${PROFILE_SAMPLE_LIMIT:-200000}"
NUM_SHARDS="${NUM_SHARDS:-128}"
MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN="${MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN:-64}"
MAX_EVAL_POINTS_PER_ACCOUNT_VALID="${MAX_EVAL_POINTS_PER_ACCOUNT_VALID:-32}"
MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION="${MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION:-32}"
MAX_EVENTS="${MAX_EVENTS:-256}"
HISTORY_TIME_ANCHOR="${HISTORY_TIME_ANCHOR:-last_event}"
INACTIVITY_PROFILE_COL="${INACTIVITY_PROFILE_COL:-seconds_since_last_event}"

mkdir -p "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/build_pragma_c_canonical_transactions.py \
  --raw_dir "${RAW_DIR}" \
  --raw_csv "${RAW_CSV}" \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/build_pragma_c_eval_points.py \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/assign_pragma_c_splits.py \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/build_pragma_c_tokenizer.py \
  --output_root "${WORK_ROOT}" \
  --max_history_events "${MAX_HISTORY_EVENTS}" \
  --profile_sample_limit "${PROFILE_SAMPLE_LIMIT}" \
  --max_events "${MAX_EVENTS}" \
  --history_time_anchor "${HISTORY_TIME_ANCHOR}" \
  --inactivity_profile_col "${INACTIVITY_PROFILE_COL}"

"${PYTHON_BIN}" scripts/audit_pragma_c_leakage.py \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/audit_pragma_c_graph.py \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/build_pragma_c_encode_index.py \
  --output_root "${WORK_ROOT}" \
  --num_shards "${NUM_SHARDS}" \
  --max_eval_points_per_account_train "${MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN}" \
  --max_eval_points_per_account_valid "${MAX_EVAL_POINTS_PER_ACCOUNT_VALID}" \
  --max_eval_points_per_account_calibration "${MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION}"
