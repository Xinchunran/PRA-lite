#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RAW_DIR="${RAW_DIR:-data/raw/ibm_aml}"
RAW_CSV="${RAW_CSV:-}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/ibm_aml_full}"
SPLIT_DIR="${SPLIT_DIR:-data/splits/ibm_aml_full}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROCESSED_DIR}/tokenizer}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${PROCESSED_DIR}/tokenized}"
TOKENIZED_BACKEND="${TOKENIZED_BACKEND:-lmdb}"
MAX_EVENTS="${MAX_EVENTS:-512}"
MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
TOKENIZE_NUM_WORKERS="${TOKENIZE_NUM_WORKERS:-$(nproc)}"
RAW_CHUNKSIZE="${RAW_CHUNKSIZE:-500000}"
RAW_SAMPLE_FRAC="${RAW_SAMPLE_FRAC:-1.0}"
RAW_SAMPLE_SEED="${RAW_SAMPLE_SEED:-42}"

CMD=(python tools/convert_ibm_aml_to_pralite.py --raw_dir "${RAW_DIR}" --processed_dir "${PROCESSED_DIR}")
if [[ -n "${RAW_CSV}" ]]; then
  CMD+=(--raw_csv "${RAW_CSV}")
fi
CMD+=(--chunksize "${RAW_CHUNKSIZE}" --sample_frac "${RAW_SAMPLE_FRAC}" --seed "${RAW_SAMPLE_SEED}")

start_ts="$(date +%s)"
echo "[prepare_ibm_aml] RAW_DIR=${RAW_DIR}"
echo "[prepare_ibm_aml] RAW_CSV=${RAW_CSV:-auto}"
echo "[prepare_ibm_aml] PROCESSED_DIR=${PROCESSED_DIR}"
echo "[prepare_ibm_aml] SPLIT_DIR=${SPLIT_DIR}"
echo "[prepare_ibm_aml] TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "[prepare_ibm_aml] TOKENIZED_DIR=${TOKENIZED_DIR}"
echo "[prepare_ibm_aml] RAW_CHUNKSIZE=${RAW_CHUNKSIZE}"
echo "[prepare_ibm_aml] RAW_SAMPLE_FRAC=${RAW_SAMPLE_FRAC}"
echo "[prepare_ibm_aml] TOKENIZE_NUM_WORKERS=${TOKENIZE_NUM_WORKERS}"
echo "[prepare_ibm_aml] stage=convert"
"${CMD[@]}"

echo "[prepare_ibm_aml] stage=split"
python tools/make_entity_splits.py \
  --labels "${PROCESSED_DIR}/labels.parquet" \
  --out_dir "${SPLIT_DIR}"

echo "[prepare_ibm_aml] stage=build_vocab"
python -m src.tokenizer.build_vocab \
  --processed_dir "${PROCESSED_DIR}" \
  --output_dir "${TOKENIZER_DIR}"

echo "[prepare_ibm_aml] stage=encode"
python -m src.tokenizer.encode_dataset \
  --processed_dir "${PROCESSED_DIR}" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --split_dir "${SPLIT_DIR}" \
  --output_dir "${TOKENIZED_DIR}" \
  --backend "${TOKENIZED_BACKEND}" \
  --max_events "${MAX_EVENTS}" \
  --max_event_tokens "${MAX_EVENT_TOKENS}" \
  --max_profile_tokens "${MAX_PROFILE_TOKENS}" \
  --num_workers "${TOKENIZE_NUM_WORKERS}"

python - <<PY
from pathlib import Path
import numpy as np

tokenized_dir = Path("${TOKENIZED_DIR}")
backend = "${TOKENIZED_BACKEND}"
print("tokenized_dir:", tokenized_dir.resolve())
print("backend:", backend)
print("parquet exists:", (tokenized_dir / "dataset.parquet").exists())
print("lmdb exists:", (tokenized_dir / "dataset.lmdb").exists())
if backend in {"lmdb", "both"}:
    entity_ids = np.load(tokenized_dir / "dataset.lmdb" / "entity_ids.npy")
    print("lmdb rows=", len(entity_ids))
PY

end_ts="$(date +%s)"
echo "[prepare_ibm_aml] done elapsed_s=$((end_ts - start_ts))"
