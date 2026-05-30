#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${RAW_DIR:-$ROOT/data/raw/ibm_aml}"
KAGGLE_DATASET="${KAGGLE_DATASET:-ealtman2019/ibm-transactions-for-anti-money-laundering-aml}"
KAGGLE_FILE="${KAGGLE_FILE:-}"

mkdir -p "${RAW_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it first, e.g.:" >&2
  echo "  conda activate pragma-lite && python -m pip install --no-cache-dir kaggle" >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Missing ${HOME}/.kaggle/kaggle.json. Configure Kaggle API token first." >&2
  exit 1
fi

CMD=(
  kaggle datasets download
  -d "${KAGGLE_DATASET}"
  -p "${RAW_DIR}"
  --unzip
)
if [[ -n "${KAGGLE_FILE}" ]]; then
  CMD+=(-f "${KAGGLE_FILE}")
fi
"${CMD[@]}"

echo "Downloaded IBM AML raw data to ${RAW_DIR}"
find "${RAW_DIR}" -maxdepth 3 -type f | sort | head -50
