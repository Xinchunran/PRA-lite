#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || { cd "${SCRIPT_DIR}/.." && pwd; })"
cd "${PROJECT_ROOT}"

export DATASET="${DATASET:-transxion_full}"
export TOKENIZED_BACKEND="${TOKENIZED_BACKEND:-lmdb}"
export FORCE_REBUILD_TOKENIZER="${FORCE_REBUILD_TOKENIZER:-0}"
export FORCE_REENCODE="${FORCE_REENCODE:-1}"
export DO_TRAIN=0

bash run.sh
