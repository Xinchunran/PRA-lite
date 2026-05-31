#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || { cd "${SCRIPT_DIR}/.." && pwd; })"
cd "${PROJECT_ROOT}"

ACTION="${1:-help}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/benchmarks/run_ibm_aml_benchmark.sh <action>

Actions:
  download   Download IBM AML Kaggle data
  prepare    Convert raw CSV -> processed -> tokenizer -> LMDB
  train      Launch 4-GPU bf16 LMDB pretraining
  all        Run prepare then train

Examples:
  bash scripts/benchmarks/run_ibm_aml_benchmark.sh download
  RAW_CSV=LI-Small_Trans.csv bash scripts/benchmarks/run_ibm_aml_benchmark.sh prepare
  TRAIN_BATCH_SIZE=32 NPROC_PER_NODE=4 bash scripts/benchmarks/run_ibm_aml_benchmark.sh train
EOF
}

case "${ACTION}" in
  help|-h|--help)
    usage
    ;;
  download)
    bash scripts/download/download_ibm_aml_kaggle.sh
    ;;
  prepare)
    bash scripts/prepare/streaming/prepare_ibm_aml_lmdb.sh
    ;;
  train)
    bash scripts/train/train_ibm_aml_lmdb.sh
    ;;
  all)
    bash scripts/prepare/streaming/prepare_ibm_aml_lmdb.sh
    bash scripts/train/train_ibm_aml_lmdb.sh
    ;;
  *)
    echo "Unsupported action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
