from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir, write_json


def split_ibm_aml_csv(raw_csv: str | Path, output_dir: str | Path, rows_per_shard: int = 250000) -> Path:
    raw_path = Path(raw_csv)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw CSV: {raw_path}")
    out_dir = ensure_dir(output_dir)
    rows_per_shard = max(int(rows_per_shard), 1)

    shard_paths: list[str] = []
    total_rows = 0
    for shard_idx, chunk in enumerate(pd.read_csv(raw_path, chunksize=rows_per_shard), start=0):
        shard_path = out_dir / f"shard_{shard_idx:05d}.csv"
        chunk.to_csv(shard_path, index=False)
        shard_paths.append(str(shard_path))
        total_rows += len(chunk)
        print(
            f"[split_ibm_aml_csv] shard={shard_idx:05d} rows={len(chunk)} total_rows={total_rows}",
            flush=True,
        )

    write_json(
        out_dir / "split_manifest.json",
        {
            "raw_csv": str(raw_path.resolve()),
            "rows_per_shard": rows_per_shard,
            "num_shards": len(shard_paths),
            "total_rows": total_rows,
            "shards": shard_paths,
        },
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_shard", type=int, default=250000)
    args = parser.parse_args()
    split_ibm_aml_csv(args.raw_csv, args.output_dir, rows_per_shard=args.rows_per_shard)


if __name__ == "__main__":
    main()
