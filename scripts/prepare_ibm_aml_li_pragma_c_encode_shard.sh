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

SHARD_INDEX="${1:-${SHARD_INDEX:-}}"
if [[ -z "${SHARD_INDEX}" ]]; then
  echo "Usage: $0 <shard_index>" >&2
  exit 1
fi

WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium_pragma_c}"
NUM_SHARDS="${NUM_SHARDS:-128}"
MAX_EVENTS="${MAX_EVENTS:-512}"
MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
MAX_HISTORY_EVENTS="${MAX_HISTORY_EVENTS:-6500}"
MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN="${MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN:-64}"
MAX_EVAL_POINTS_PER_ACCOUNT_VALID="${MAX_EVAL_POINTS_PER_ACCOUNT_VALID:-32}"
MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION="${MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION:-32}"
MANIFEST_LOCK="${WORK_ROOT}/.manifest.lock"

echo "[pragma_c_encode_wrapper] shard=${SHARD_INDEX} stage=encode start" >&2
"${PYTHON_BIN}" scripts/encode_pragma_c_records.py \
  --output_root "${WORK_ROOT}" \
  --shard_index "${SHARD_INDEX}" \
  --num_shards "${NUM_SHARDS}" \
  --max_events "${MAX_EVENTS}" \
  --max_event_tokens "${MAX_EVENT_TOKENS}" \
  --max_profile_tokens "${MAX_PROFILE_TOKENS}" \
  --max_history_events "${MAX_HISTORY_EVENTS}" \
  --max_eval_points_per_account_train "${MAX_EVAL_POINTS_PER_ACCOUNT_TRAIN}" \
  --max_eval_points_per_account_valid "${MAX_EVAL_POINTS_PER_ACCOUNT_VALID}" \
  --max_eval_points_per_account_calibration "${MAX_EVAL_POINTS_PER_ACCOUNT_CALIBRATION}"

echo "[pragma_c_encode_wrapper] shard=${SHARD_INDEX} stage=manifest start" >&2
mkdir -p "$(dirname "${MANIFEST_LOCK}")"
(
  flock 9
  "${PYTHON_BIN}" scripts/build_pragma_c_manifest.py \
    --output_root "${WORK_ROOT}"
) 9>"${MANIFEST_LOCK}"
echo "[pragma_c_encode_wrapper] shard=${SHARD_INDEX} stage=done" >&2
