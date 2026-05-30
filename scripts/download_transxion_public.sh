#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw/transxion_public"
TMP_DIR="$ROOT/data/raw/_tmp_transxion_repo"

mkdir -p "$ROOT/data/raw"
rm -rf "$TMP_DIR"

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/chaos-max/TransXion.git "$TMP_DIR"
cd "$TMP_DIR"
git lfs pull

mkdir -p "$RAW_DIR"
cp data/person.csv "$RAW_DIR/person.csv"
cp data/merchant.csv "$RAW_DIR/merchant.csv"
cp data/tx.csv "$RAW_DIR/tx.csv"

cd "$ROOT"
rm -rf "$TMP_DIR"

echo "Downloaded TransXion public data to $RAW_DIR"
ls -lh "$RAW_DIR"
