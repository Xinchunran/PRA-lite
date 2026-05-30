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

RAW_DIR="${RAW_DIR:-data/raw/ibm_aml}"
RAW_CSV="${RAW_CSV:-LI-Medium_Trans.csv}"
WORK_ROOT="${WORK_ROOT:-data/streaming/ibm_aml_li_medium}"
RAW_SHARD_DIR="${RAW_SHARD_DIR:-${WORK_ROOT}/raw_shards}"
PROCESSED_SHARD_ROOT="${PROCESSED_SHARD_ROOT:-${WORK_ROOT}/processed_shards}"
TOKENIZED_SHARD_ROOT="${TOKENIZED_SHARD_ROOT:-${WORK_ROOT}/tokenized_shards}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_ROOT}/tokenizer}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifest.json}"
SPLIT_DIR="${SPLIT_DIR:-${WORK_ROOT}/split_stub}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_ibm_aml_medium_streaming}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train.log}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}/plots}"
AUTO_RESUME="${AUTO_RESUME:-1}"

ROWS_PER_SHARD="${ROWS_PER_SHARD:-250000}"
HASH_SPLIT_SEED="${HASH_SPLIT_SEED:-26}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
VALID_FRAC="${VALID_FRAC:-0.1}"
MAX_EVENTS="${MAX_EVENTS:-512}"
MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
TOKENIZE_NUM_WORKERS="${TOKENIZE_NUM_WORKERS:-8}"

CONFIG="${CONFIG:-configs/train/pretrain_mlm_small.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRECISION="${PRECISION:-bf16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
ENABLE_TF32="${ENABLE_TF32:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RAW_SHARD_DIR}" "${PROCESSED_SHARD_ROOT}" "${TOKENIZED_SHARD_ROOT}" "${TOKENIZER_DIR}" "${SPLIT_DIR}" "${OUTPUT_DIR}" "${PLOTS_DIR}"

resolve_raw_csv() {
  if [[ -f "${RAW_CSV}" ]]; then
    "${PYTHON_BIN}" - <<PY
from pathlib import Path
print(Path("${RAW_CSV}").resolve())
PY
    return
  fi
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
raw_dir = Path("${RAW_DIR}")
candidate = raw_dir / "${RAW_CSV}"
if not candidate.exists():
    raise SystemExit(f"Missing raw CSV: {candidate}")
print(candidate.resolve())
PY
}

append_manifest() {
  local shard_name="$1"
  local tokenized_dir="$2"
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
print("[streaming_manifest] added ${shard_name}")
PY
}

prepare_shard() {
  local shard_csv="$1"
  local build_vocab_flag="$2"
  local shard_name
  shard_name="$(basename "${shard_csv}" .csv)"
  local processed_dir="${PROCESSED_SHARD_ROOT}/${shard_name}"
  local tokenized_dir="${TOKENIZED_SHARD_ROOT}/${shard_name}"
  local reuse_ok="0"

  if [[ -f "${tokenized_dir}/dataset.lmdb/length.txt" && -f "${tokenized_dir}/train.lmdb/length.txt" && -f "${tokenized_dir}/valid.lmdb/length.txt" ]]; then
    if [[ -f "${tokenized_dir}/tokenized_summary.json" && -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
      reuse_ok="$(
        "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary_path = Path("${tokenized_dir}/tokenized_summary.json")
tokenizer_path = Path("${TOKENIZER_DIR}/tokenizer.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
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
    echo "[streaming_prepare] shard=${shard_name} reuse tokenized ${tokenized_dir}"
    append_manifest "${shard_name}" "${tokenized_dir}"
    return
  fi

  if [[ -d "${tokenized_dir}" ]]; then
    echo "[streaming_prepare] shard=${shard_name} removing incomplete tokenized dir ${tokenized_dir}"
    rm -rf "${tokenized_dir}"
  fi

  echo "[streaming_prepare] shard=${shard_name} stage=convert"
  "${PYTHON_BIN}" tools/convert_ibm_aml_to_pralite.py \
    --raw_dir "${RAW_DIR}" \
    --raw_csv "${shard_csv}" \
    --processed_dir "${processed_dir}"

  if [[ "${build_vocab_flag}" == "1" || ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "[streaming_prepare] shard=${shard_name} stage=build_vocab"
    "${PYTHON_BIN}" -m src.tokenizer.build_vocab \
      --processed_dir "${processed_dir}" \
      --output_dir "${TOKENIZER_DIR}"
  fi

  echo "[streaming_prepare] shard=${shard_name} stage=encode"
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

  append_manifest "${shard_name}" "${tokenized_dir}"
}

echo "[streaming_run] PYTHON_BIN=${PYTHON_BIN}"
echo "[streaming_run] WORK_ROOT=${WORK_ROOT}"
echo "[streaming_run] mode=train_only"

# Training-only mode: intentionally skip raw split / preprocess / tokenization.
DATA_DIR_RESOLVED="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest_path = Path("${MANIFEST_PATH}")
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("shards", []):
        if str(entry.get("status")) == "ready":
            tokenized_dir = entry.get("tokenized_dir")
            if tokenized_dir:
                print(Path(tokenized_dir).resolve())
                raise SystemExit(0)

tokenized_root = Path("${TOKENIZED_SHARD_ROOT}")
for candidate in sorted(tokenized_root.glob("shard_*")):
    if (candidate / "train.lmdb" / "length.txt").exists():
        print(candidate.resolve())
        raise SystemExit(0)

raise SystemExit("No ready tokenized shard found. Populate manifest.json or tokenized_shards first.")
PY
)"
SHARD_NAME="$(basename "${DATA_DIR_RESOLVED}")"
PREPARE_EXIT_CODE=0

echo "[streaming_run] DATA_DIR=${DATA_DIR_RESOLVED}"
echo "[streaming_run] SHARD_NAME=${SHARD_NAME}"

export CONFIG MODEL_CONFIG MANIFEST_PATH OUTPUT_DIR TOKENIZER_DIR SPLIT_DIR
export NPROC_PER_NODE PRECISION TRAIN_BATCH_SIZE DATALOADER_NUM_WORKERS
export DATALOADER_PREFETCH_FACTOR DATALOADER_PERSISTENT_WORKERS DATALOADER_PIN_MEMORY
export ENABLE_TF32 PYTORCH_CUDA_ALLOC_CONF
export DATA_DIR="${DATA_DIR_RESOLVED}"
export PROCESSED_DIR="${PROCESSED_SHARD_ROOT}/${SHARD_NAME}"
if [[ "${AUTO_RESUME}" == "1" && -f "${OUTPUT_DIR}/last.ckpt" ]]; then
  export RESUME_FROM="${OUTPUT_DIR}/last.ckpt"
  echo "[streaming_run] RESUME_FROM=${RESUME_FROM}"
fi

echo "[streaming_run] stage=train"
echo "[streaming_run] TRAIN_LOG=${TRAIN_LOG}"
echo "[streaming_run] PLOTS_DIR=${PLOTS_DIR}"
set +e
CHECK_SPLITS=0 bash scripts/train_pretrain_ddp.sh 2>&1 | tee -a "${TRAIN_LOG}"
TRAIN_EXIT_CODE=$?
set -e

if [[ -f "${OUTPUT_DIR}/metrics.jsonl" ]]; then
  "${PYTHON_BIN}" tools/plot_pretrain_metrics.py \
    --metrics_file "${OUTPUT_DIR}/metrics.jsonl" \
    --output_dir "${PLOTS_DIR}" \
    --title_prefix "IBM AML Medium Streaming" || true
fi

echo "[streaming_run] train_exit=${TRAIN_EXIT_CODE} prepare_exit=${PREPARE_EXIT_CODE}"
exit "${TRAIN_EXIT_CODE}"
