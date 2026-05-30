#!/bin/bash
set -euo pipefail

DATASET="${1:-all}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${ROOT}/data/raw"
mkdir -p "${RAW_ROOT}"

python - <<PY
from pathlib import Path
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

dataset = "${DATASET}"
raw_root = Path("${RAW_ROOT}")
raw_root.mkdir(parents=True, exist_ok=True)

# Public mirrors normally do not need auth, but allow HF_TOKEN override.
hf_token = os.environ.get("HF_TOKEN", None)
token_arg = hf_token if hf_token else None


def save_datasetdict_to_parquet(repo_id: str, out_dir: Path):
    print(f"\\n=== Loading {repo_id} via datasets.load_dataset ===")
    ds = load_dataset(repo_id, token=token_arg)
    print(ds)

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in ds.items():
        print(f"split={split_name}")
        print(f"rows={len(split_ds)}")
        print(f"columns={split_ds.column_names}")

        out_path = out_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        print(f"saved {out_path}")

    return ds


def download_paysim():
    out_dir = raw_root / "paysim_hf"
    repo_id = "theman10/paysim"

    ds = save_datasetdict_to_parquet(repo_id, out_dir)

    total_rows = sum(len(v) for v in ds.values())
    print(f"PaySim total rows={total_rows}")

    expected_cols = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }

    first_split = next(iter(ds.values()))
    cols = set(first_split.column_names)
    missing = sorted(expected_cols - cols)

    if missing:
        raise SystemExit(f"PaySim missing expected columns: {missing}")

    if total_rows < 6_000_000:
        raise SystemExit(f"PaySim row count looks too small: {total_rows}")

    print("PaySim HF download/schema check passed")


def download_elliptic():
    out_dir = raw_root / "elliptic_hf"
    repo_id = "asjoie/elliptic-bitcoin-dataset"

    try:
        ds = save_datasetdict_to_parquet(repo_id, out_dir)

        total_rows = sum(len(v) for v in ds.values())
        print(f"Elliptic total rows from datasets={total_rows}")

        if total_rows <= 0:
            raise SystemExit("Elliptic loaded zero rows")

        print("Elliptic HF load_dataset check passed")
        return
    except Exception as e:
        print("\\nload_dataset failed for Elliptic.")
        print("Falling back to huggingface_hub.snapshot_download.")
        print("Original error:", repr(e))

    snapshot_dir = out_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        token=token_arg,
    )

    print(f"snapshot downloaded to: {path}")

    files = sorted([p for p in snapshot_dir.rglob("*") if p.is_file()])
    print("Downloaded files:")
    for p in files[:100]:
        print(p, p.stat().st_size)

    if not files:
        raise SystemExit("Elliptic snapshot download produced no files")

    print("Elliptic HF snapshot_download check passed")


if dataset in ("paysim", "all"):
    download_paysim()

if dataset in ("elliptic", "all"):
    download_elliptic()

if dataset not in ("paysim", "elliptic", "all"):
    raise SystemExit("Usage: bash scripts/download_hf_public_datasets.sh [all|paysim|elliptic]")
PY

echo
echo "Downloaded files under ${RAW_ROOT}:"
python - <<PY
from pathlib import Path

raw_root = Path("${RAW_ROOT}")
files = sorted([p for p in raw_root.rglob("*") if p.is_file()], key=lambda p: str(p))
for path in files[-100:]:
    print(f"{path} {path.stat().st_size}")
PY
