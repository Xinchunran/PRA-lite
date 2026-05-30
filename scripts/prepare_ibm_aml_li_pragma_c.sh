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
WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_c}"
MAX_HISTORY_EVENTS="${MAX_HISTORY_EVENTS:-6500}"
PROFILE_SAMPLE_LIMIT="${PROFILE_SAMPLE_LIMIT:-200000}"

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
  --profile_sample_limit "${PROFILE_SAMPLE_LIMIT}"

"${PYTHON_BIN}" scripts/audit_pragma_c_leakage.py \
  --output_root "${WORK_ROOT}"

"${PYTHON_BIN}" scripts/audit_pragma_c_graph.py \
  --output_root "${WORK_ROOT}"
