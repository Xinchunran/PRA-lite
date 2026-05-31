#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import write_json


def audit_graph(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    eval_points = pd.read_parquet(output_root / "eval_points" / "eval_points.parquet")
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")

    payload: dict[str, object] = {
        "account_overlap": {},
        "split_stats": {},
    }
    split_entities = {
        split_name: set(df["entity_id"].astype("int64").tolist())
        for split_name, df in eval_points.groupby("split", sort=False)
    }
    split_names = sorted(split_entities)
    for split_name in split_names:
        split_df = eval_points[eval_points["split"] == split_name]
        payload["split_stats"][split_name] = {
            "num_eval_points": int(len(split_df)),
            "num_unique_entities": int(split_df["entity_id"].nunique()),
            "num_positive_labels": int(pd.to_numeric(split_df["label"], errors="coerce").fillna(0).sum()),
        }
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlap = len(split_entities[left].intersection(split_entities[right]))
            payload["account_overlap"][f"{left}__{right}"] = overlap

    write_json(output_root / "graph_audit.json", payload)
    return output_root / "graph_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    args = parser.parse_args()
    audit_graph(args.output_root)


if __name__ == "__main__":
    main()
