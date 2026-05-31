#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw/paysim"
mkdir -p "$RAW_DIR"

kaggle datasets download -d ealaxi/paysim1 -p "$RAW_DIR" --unzip

if [ -f "$RAW_DIR/PS_20174392719_1491204439457_log.csv" ]; then
  cp "$RAW_DIR/PS_20174392719_1491204439457_log.csv" "$RAW_DIR/transactions.csv"
fi

echo "Downloaded PaySim to $RAW_DIR"
ls -lh "$RAW_DIR"
