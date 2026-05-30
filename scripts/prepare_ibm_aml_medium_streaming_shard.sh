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

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Cannot resolve python interpreter. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

SHARD_SPEC="${1:-${SHARD_SPEC:-}}"
if [[ -z "${SHARD_SPEC}" ]]; then
  echo "Usage: $0 <shard_index|shard_name|/abs/path/to/shard.csv>" >&2
  exit 1
fi

RAW_DIR="${RAW_DIR:-data/raw/ibm_aml}"
WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium}"
RAW_SHARD_DIR="${RAW_SHARD_DIR:-${WORK_ROOT}/raw_shards}"
PROCESSED_SHARD_ROOT="${PROCESSED_SHARD_ROOT:-${WORK_ROOT}/processed_shards}"
TOKENIZED_SHARD_ROOT="${TOKENIZED_SHARD_ROOT:-${WORK_ROOT}/tokenized_shards}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_ROOT}/tokenizer}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifest.json}"

HASH_SPLIT_SEED="${HASH_SPLIT_SEED:-26}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
VALID_FRAC="${VALID_FRAC:-0.1}"
MAX_EVENTS="${MAX_EVENTS:-256}"
MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
TOKENIZE_NUM_WORKERS="${TOKENIZE_NUM_WORKERS:-$(nproc)}"

mkdir -p "${PROCESSED_SHARD_ROOT}" "${TOKENIZED_SHARD_ROOT}"

resolve_shard_csv() {
  local spec="$1"
  if [[ -f "${spec}" ]]; then
    "${PYTHON_BIN}" - <<PY
from pathlib import Path
print(Path("${spec}").resolve())
PY
    return
  fi
  if [[ "${spec}" =~ ^[0-9]+$ ]]; then
    printf "%s/%s\n" "${RAW_SHARD_DIR}" "shard_$(printf '%05d' "${spec}").csv"
    return
  fi
  if [[ "${spec}" == shard_*.csv ]]; then
    printf "%s/%s\n" "${RAW_SHARD_DIR}" "${spec}"
    return
  fi
  if [[ "${spec}" == shard_* ]]; then
    printf "%s/%s.csv\n" "${RAW_SHARD_DIR}" "${spec}"
    return
  fi
  printf "%s/%s\n" "${RAW_SHARD_DIR}" "${spec}"
}

append_manifest_locked() {
  local shard_name="$1"
  local tokenized_dir="$2"
  local lock_path="${MANIFEST_PATH}.lock"
  mkdir -p "$(dirname "${MANIFEST_PATH}")"
  : > "${lock_path}"
  exec 9>"${lock_path}"
  flock -x 9
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest_path = Path("${MANIFEST_PATH}")
manifest = {"tokenizer_dir": str(Path("${TOKENIZER_DIR}").resolve()), "shards": []}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
shards = [entry for entry in manifest.get("shards", []) if str(entry.get("name")) != "${shard_name}"]
shards.append(
    {
        "name": "${shard_name}",
        "tokenized_dir": str(Path("${tokenized_dir}").resolve()),
        "status": "ready",
    }
)
manifest["tokenizer_dir"] = str(Path("${TOKENIZER_DIR}").resolve())
manifest["shards"] = sorted(shards, key=lambda item: str(item.get("name", "")))
tmp_path = manifest_path.with_suffix(".tmp")
tmp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
tmp_path.replace(manifest_path)
print(f"[streaming_manifest] added ${shard_name}")
PY
  flock -u 9
}

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
  echo "Missing tokenizer at ${TOKENIZER_DIR}/tokenizer.json" >&2
  exit 1
fi

shard_csv="$(resolve_shard_csv "${SHARD_SPEC}")"
if [[ ! -f "${shard_csv}" ]]; then
  echo "Missing shard CSV: ${shard_csv}" >&2
  exit 1
fi

shard_name="$(basename "${shard_csv}" .csv)"
processed_dir="${PROCESSED_SHARD_ROOT}/${shard_name}"
tokenized_dir="${TOKENIZED_SHARD_ROOT}/${shard_name}"
reuse_ok="0"

echo "[prepare_shard] shard=${shard_name}"
echo "[prepare_shard] shard_csv=${shard_csv}"
echo "[prepare_shard] processed_dir=${processed_dir}"
echo "[prepare_shard] tokenized_dir=${tokenized_dir}"
echo "[prepare_shard] TOKENIZE_NUM_WORKERS=${TOKENIZE_NUM_WORKERS}"

if [[ -f "${tokenized_dir}/dataset.lmdb/length.txt" && -f "${tokenized_dir}/train.lmdb/length.txt" && -f "${tokenized_dir}/valid.lmdb/length.txt" ]]; then
  if [[ -f "${tokenized_dir}/tokenized_summary.json" ]]; then
    reuse_ok="$(
      "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary = json.loads(Path("${tokenized_dir}/tokenized_summary.json").read_text(encoding="utf-8"))
tokenizer = json.loads(Path("${TOKENIZER_DIR}/tokenizer.json").read_text(encoding="utf-8"))
expected = {
    "vocab_size": len(tokenizer.get("token_to_id", {})),
    "max_events": int("${MAX_EVENTS}"),
    "max_event_tokens": int("${MAX_EVENT_TOKENS}"),
    "max_profile_tokens": int("${MAX_PROFILE_TOKENS}"),
}
actual = {
    "vocab_size": int(summary.get("vocab_size", -1)),
    "max_events": int(summary.get("max_events", -1)),
    "max_event_tokens": int(summary.get("max_event_tokens", -1)),
    "max_profile_tokens": int(summary.get("max_profile_tokens", -1)),
}
print("1" if actual == expected else "0")
PY
    )"
  fi
fi

if [[ "${reuse_ok}" == "1" ]]; then
  echo "[prepare_shard] shard=${shard_name} reuse existing tokenized data"
  append_manifest_locked "${shard_name}" "${tokenized_dir}"
  exit 0
fi

if [[ -d "${tokenized_dir}" ]]; then
  echo "[prepare_shard] shard=${shard_name} removing stale tokenized dir"
  rm -rf "${tokenized_dir}"
fi

echo "[prepare_shard] shard=${shard_name} stage=convert"
"${PYTHON_BIN}" tools/convert_ibm_aml_to_pralite.py \
  --raw_dir "${RAW_DIR}" \
  --raw_csv "${shard_csv}" \
  --processed_dir "${processed_dir}"

echo "[prepare_shard] shard=${shard_name} stage=encode"
"${PYTHON_BIN}" -m src.tokenizer.encode_dataset \
  --processed_dir "${processed_dir}" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --output_dir "${tokenized_dir}" \
  --backend lmdb \
  --max_events "${MAX_EVENTS}" \
  --max_event_tokens "${MAX_EVENT_TOKENS}" \
  --max_profile_tokens "${MAX_PROFILE_TOKENS}" \
  --num_workers "${TOKENIZE_NUM_WORKERS}" \
  --hash_split_seed "${HASH_SPLIT_SEED}" \
  --train_frac "${TRAIN_FRAC}" \
  --valid_frac "${VALID_FRAC}"

append_manifest_locked "${shard_name}" "${tokenized_dir}"
echo "[prepare_shard] shard=${shard_name} done"
