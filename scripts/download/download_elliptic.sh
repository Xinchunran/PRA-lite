#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw/elliptic"
mkdir -p "$RAW_DIR"

kaggle datasets download -d ellipticco/elliptic-data-set -p "$RAW_DIR" --unzip

echo "Downloaded Elliptic to $RAW_DIR"
find "$RAW_DIR" -maxdepth 3 -type f -print
