#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw/fdb"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/fraud-dataset-benchmark" ]; then
  git clone https://github.com/amazon-science/fraud-dataset-benchmark.git "$RAW_DIR/fraud-dataset-benchmark"
else
  cd "$RAW_DIR/fraud-dataset-benchmark"
  git pull
fi

echo "FDB repo available at $RAW_DIR/fraud-dataset-benchmark"
