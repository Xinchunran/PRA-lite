#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.pragma_c.common import PRETRAIN_EVAL_SOURCES, apply_split_caps, stable_hash_bucket


def build_encode_index(
    output_root: str | Path,
    *,
    num_shards: int,
    max_eval_points_per_account_train: int,
    max_eval_points_per_account_valid: int,
    max_eval_points_per_account_calibration: int,
) -> Path:
    output_root = Path(output_root)
    started_at = time.perf_counter()
    eval_path = output_root / "eval_points" / "eval_points.parquet"
    shard_dir = ensure_dir(output_root / "eval_points" / "encode_shards")

    print(f"[pragma_c_index] loading eval points from {eval_path}", flush=True)
    eval_points = pd.read_parquet(
        eval_path,
        columns=[
            "eval_id",
            "task",
            "eval_source",
            "entity_id",
            "evaluation_time",
            "split",
            "label",
            "anchor_transaction_id",
        ],
    )
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    eval_points = eval_points[
        (eval_points["task"] == "pretrain") & (eval_points["eval_source"].isin(PRETRAIN_EVAL_SOURCES))
    ].copy()
    print(f"[pragma_c_index] pretrain eval rows={len(eval_points)}", flush=True)

    split_caps = {
        "train": max_eval_points_per_account_train,
        "valid": max_eval_points_per_account_valid,
        "calibration": max_eval_points_per_account_calibration,
        "test": 0,
        "embargo": 0,
    }
    eval_points = apply_split_caps(eval_points, split_caps).reset_index(drop=True)
    print(f"[pragma_c_index] capped eval rows={len(eval_points)}", flush=True)

    eval_points["encode_shard_index"] = eval_points["entity_id"].map(lambda value: stable_hash_bucket(value, num_shards))
    summary_rows: list[dict[str, int | str]] = []
    for shard_index, shard_eval in eval_points.groupby("encode_shard_index", sort=True):
        shard_path = shard_dir / f"shard_{int(shard_index):05d}.parquet"
        shard_eval.sort_values(["entity_id", "evaluation_time", "anchor_transaction_id"], kind="stable").to_parquet(
            shard_path,
            index=False,
        )
        summary_rows.append(
            {
                "shard_index": int(shard_index),
                "num_eval_points": int(len(shard_eval)),
                "num_unique_entities": int(shard_eval["entity_id"].nunique()),
                "path": str(shard_path.resolve()),
            }
        )
        if len(summary_rows) % 16 == 0:
            print(
                f"[pragma_c_index] wrote {len(summary_rows)}/{num_shards} shard eval files elapsed_s={time.perf_counter() - started_at:.1f}",
                flush=True,
            )

    write_json(
        output_root / "eval_points" / "encode_index_summary.json",
        {
            "num_shards": int(num_shards),
            "num_shard_files": int(len(summary_rows)),
            "rows": summary_rows,
            "elapsed_s": round(time.perf_counter() - started_at, 2),
        },
    )
    print(
        f"[pragma_c_index] complete shard_files={len(summary_rows)} elapsed_s={time.perf_counter() - started_at:.1f}",
        flush=True,
    )
    return shard_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    parser.add_argument("--num_shards", type=int, default=128)
    parser.add_argument("--max_eval_points_per_account_train", type=int, default=64)
    parser.add_argument("--max_eval_points_per_account_valid", type=int, default=32)
    parser.add_argument("--max_eval_points_per_account_calibration", type=int, default=32)
    args = parser.parse_args()
    build_encode_index(
        args.output_root,
        num_shards=args.num_shards,
        max_eval_points_per_account_train=args.max_eval_points_per_account_train,
        max_eval_points_per_account_valid=args.max_eval_points_per_account_valid,
        max_eval_points_per_account_calibration=args.max_eval_points_per_account_calibration,
    )


if __name__ == "__main__":
    main()
