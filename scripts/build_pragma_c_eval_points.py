#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir, write_json


def build_eval_points(
    output_root: str | Path,
    *,
    include_downstream: bool = True,
) -> Path:
    output_root = Path(output_root)
    canonical_path = output_root / "canonical" / "transactions.parquet"
    eval_dir = ensure_dir(output_root / "eval_points")

    tx = pd.read_parquet(canonical_path)
    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")

    sender_eval = pd.DataFrame(
        {
            "task": "pretrain",
            "eval_source": "transaction_sender",
            "entity_id": tx["sender_entity_id"].astype("int64"),
            "evaluation_time": tx["transaction_time"],
            "anchor_transaction_id": tx["transaction_id"].astype("int64"),
            "sender_entity_id": tx["sender_entity_id"].astype("int64"),
            "receiver_entity_id": tx["receiver_entity_id"].astype("int64"),
            "label": tx["is_laundering"].astype("int64"),
            "label_type": "transaction_is_laundering",
            "history_start_time": pd.NaT,
            "history_end_time": tx["transaction_time"],
        }
    )
    receiver_eval = pd.DataFrame(
        {
            "task": "pretrain",
            "eval_source": "transaction_receiver",
            "entity_id": tx["receiver_entity_id"].astype("int64"),
            "evaluation_time": tx["transaction_time"],
            "anchor_transaction_id": tx["transaction_id"].astype("int64"),
            "sender_entity_id": tx["sender_entity_id"].astype("int64"),
            "receiver_entity_id": tx["receiver_entity_id"].astype("int64"),
            "label": tx["is_laundering"].astype("int64"),
            "label_type": "transaction_is_laundering",
            "history_start_time": pd.NaT,
            "history_end_time": tx["transaction_time"],
        }
    )
    eval_frames = [sender_eval, receiver_eval]

    if include_downstream:
        pair_eval = pd.DataFrame(
            {
                "task": "downstream",
                "eval_source": "transaction_pair",
                "entity_id": tx["sender_entity_id"].astype("int64"),
                "evaluation_time": tx["transaction_time"],
                "anchor_transaction_id": tx["transaction_id"].astype("int64"),
                "sender_entity_id": tx["sender_entity_id"].astype("int64"),
                "receiver_entity_id": tx["receiver_entity_id"].astype("int64"),
                "label": tx["is_laundering"].astype("int64"),
                "label_type": "transaction_is_laundering",
                "history_start_time": pd.NaT,
                "history_end_time": tx["transaction_time"],
            }
        )
        eval_frames.append(pair_eval)

    eval_points = pd.concat(eval_frames, ignore_index=True)
    eval_points = eval_points.sort_values(
        ["evaluation_time", "anchor_transaction_id", "eval_source", "entity_id"],
        kind="stable",
    ).reset_index(drop=True)
    eval_points.insert(0, "eval_id", [f"eval_{i:012d}" for i in range(len(eval_points))])
    eval_points.to_parquet(eval_dir / "eval_points.parquet", index=False)

    summary = {
        "num_eval_points": int(len(eval_points)),
        "num_pretrain_eval_points": int((eval_points["task"] == "pretrain").sum()),
        "num_downstream_eval_points": int((eval_points["task"] == "downstream").sum()),
        "num_unique_entities": int(eval_points["entity_id"].nunique()),
        "min_evaluation_time": str(eval_points["evaluation_time"].min()),
        "max_evaluation_time": str(eval_points["evaluation_time"].max()),
    }
    write_json(eval_dir / "eval_point_sampling_summary.json", summary)
    return eval_dir / "eval_points.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    parser.add_argument("--skip_downstream", action="store_true")
    args = parser.parse_args()
    build_eval_points(args.output_root, include_downstream=not args.skip_downstream)


if __name__ == "__main__":
    main()
