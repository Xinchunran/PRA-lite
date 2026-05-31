#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || { cd "${SCRIPT_DIR}/.." && pwd; })"
cd "${PROJECT_ROOT}"

ACTION="${1:-help}"
SCALE="${2:-both}"

RAW_PUBLIC_DIR="${RAW_PUBLIC_DIR:-data/raw/transxion_public}"
RAW_CANONICAL_DIR="${RAW_CANONICAL_DIR:-data/raw/transxion_public_canonical}"
FULL_CONFIG="${FULL_CONFIG:-configs/data/transxion_full.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/pragma_lite_small.yaml}"
BACKEND="${BACKEND:-nccl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CHECK_SPLITS="${CHECK_SPLITS:-1}"
MAX_EVENTS="${MAX_EVENTS:-512}"
MAX_EVENT_TOKENS="${MAX_EVENT_TOKENS:-24}"
MAX_PROFILE_TOKENS="${MAX_PROFILE_TOKENS:-200}"
MINI_TARGET_EVENTS="${MINI_TARGET_EVENTS:-200000}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/benchmarks/run_transxion_benchmark.sh <action> <scale>

Actions:
  download   Download TransXion public raw files only
  prepare    Build processed/tokenized data and splits
  train      Launch pretraining
  all        Run prepare then train

Scales:
  mini       0.2M benchmark using transxion_200k
  small      full public benchmark using transxion_full
  both       run mini then small

Examples:
  bash scripts/benchmarks/run_transxion_benchmark.sh prepare mini
  NPROC_PER_NODE=2 bash scripts/benchmarks/run_transxion_benchmark.sh train small
  NPROC_PER_NODE=2 bash scripts/benchmarks/run_transxion_benchmark.sh all both

Important:
  - This script never downloads data unless action=download.
  - mini maps to data/processed/transxion_200k
  - small maps to data/processed/transxion_full
EOF
}

need_raw_files() {
  for file in person.csv merchant.csv tx.csv; do
    if [[ ! -f "${RAW_PUBLIC_DIR}/${file}" ]]; then
      echo "Missing raw file: ${RAW_PUBLIC_DIR}/${file}" >&2
      echo "Mount or download raw TransXion public files first, then rerun." >&2
      exit 1
    fi
  done
}

prepare_full_processed() {
  need_raw_files
  python tools/prepare_transxion_public_raw.py \
    --raw_dir "${RAW_PUBLIC_DIR}" \
    --out_dir "${RAW_CANONICAL_DIR}"

  python -m src.data_downloader.build_events \
    --config "${FULL_CONFIG}"
}

prepare_mini() {
  prepare_full_processed
  python tools/make_entity_event_cut.py \
    --processed_dir data/processed/transxion_full \
    --output_dir data/processed/transxion_200k \
    --target_events "${MINI_TARGET_EVENTS}"

  python tools/make_entity_splits.py \
    --labels data/processed/transxion_200k/labels.parquet \
    --out_dir data/splits/transxion_200k

  python -m src.tokenizer.build_vocab \
    --processed_dir data/processed/transxion_200k \
    --output_dir data/processed/transxion_200k/tokenizer

  python -m src.tokenizer.encode_dataset \
    --processed_dir data/processed/transxion_200k \
    --tokenizer_dir data/processed/transxion_200k/tokenizer \
    --output_dir data/processed/transxion_200k/tokenized \
    --max_events "${MAX_EVENTS}" \
    --max_event_tokens "${MAX_EVENT_TOKENS}" \
    --max_profile_tokens "${MAX_PROFILE_TOKENS}"
}

prepare_small() {
  prepare_full_processed
  python tools/make_entity_splits.py \
    --labels data/processed/transxion_full/labels.parquet \
    --out_dir data/splits/transxion_full

  python -m src.tokenizer.build_vocab \
    --processed_dir data/processed/transxion_full \
    --output_dir data/processed/transxion_full/tokenizer

  python -m src.tokenizer.encode_dataset \
    --processed_dir data/processed/transxion_full \
    --tokenizer_dir data/processed/transxion_full/tokenizer \
    --output_dir data/processed/transxion_full/tokenized \
    --max_events "${MAX_EVENTS}" \
    --max_event_tokens "${MAX_EVENT_TOKENS}" \
    --max_profile_tokens "${MAX_PROFILE_TOKENS}"
}

train_scale() {
  local scale="$1"
  local dataset_name=""
  local train_config=""

  case "${scale}" in
    mini)
      dataset_name="transxion_200k"
      train_config="configs/train/pretrain_mlm_mini.yaml"
      ;;
    small)
      dataset_name="transxion_full"
      train_config="configs/train/pretrain_mlm_small.yaml"
      ;;
    *)
      echo "Unsupported scale for train: ${scale}" >&2
      exit 1
      ;;
  esac

  CONFIG="${TRAIN_CONFIG:-${train_config}}" \
  MODEL_CONFIG="${MODEL_CONFIG}" \
  DATA_DIR="data/processed/${dataset_name}/tokenized" \
  SPLIT_DIR="data/splits/${dataset_name}" \
  PROCESSED_DIR="data/processed/${dataset_name}" \
  TOKENIZER_DIR="data/processed/${dataset_name}/tokenizer" \
  OUTPUT_DIR="runs/pretrain_${dataset_name}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  BACKEND="${BACKEND}" \
  CHECK_SPLITS="${CHECK_SPLITS}" \
  bash scripts/train/train_pretrain_ddp.sh
}

run_prepare() {
  case "${SCALE}" in
    mini) prepare_mini ;;
    small) prepare_small ;;
    both)
      prepare_mini
      prepare_small
      ;;
    *)
      echo "Unsupported scale: ${SCALE}" >&2
      exit 1
      ;;
  esac
}

run_train() {
  case "${SCALE}" in
    mini) train_scale mini ;;
    small) train_scale small ;;
    both)
      train_scale mini
      train_scale small
      ;;
    *)
      echo "Unsupported scale: ${SCALE}" >&2
      exit 1
      ;;
  esac
}

case "${ACTION}" in
  help|-h|--help)
    usage
    ;;
  download)
    bash scripts/download/download_transxion_public.sh
    ;;
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  all)
    run_prepare
    run_train
    ;;
  *)
    echo "Unsupported action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
