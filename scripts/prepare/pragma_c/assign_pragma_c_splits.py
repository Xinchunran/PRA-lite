#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import write_json


DEFAULT_FRACTIONS = {
    "train": 0.68,
    "embargo_1": 0.70,
    "valid": 0.82,
    "calibration": 0.87,
    "embargo_2": 0.89,
}


def _boundary_time(unique_times: pd.Series, cumulative_counts: pd.Series, target_count: int) -> pd.Timestamp:
    idx = int(cumulative_counts.searchsorted(target_count, side="left"))
    idx = min(max(idx, 0), len(unique_times) - 1)
    return pd.Timestamp(unique_times.iloc[idx]).tz_convert("UTC")


def assign_splits(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    eval_path = output_root / "eval_points" / "eval_points.parquet"
    eval_points = pd.read_parquet(eval_path)
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    eval_points = eval_points.sort_values(["evaluation_time", "eval_id"], kind="stable").reset_index(drop=True)

    counts_by_time = eval_points.groupby("evaluation_time", sort=True).size().reset_index(name="count")
    counts_by_time["cumulative"] = counts_by_time["count"].cumsum()
    total = int(len(eval_points))

    train_end = _boundary_time(counts_by_time["evaluation_time"], counts_by_time["cumulative"], int(total * DEFAULT_FRACTIONS["train"]))
    embargo_1_end = _boundary_time(counts_by_time["evaluation_time"], counts_by_time["cumulative"], int(total * DEFAULT_FRACTIONS["embargo_1"]))
    valid_end = _boundary_time(counts_by_time["evaluation_time"], counts_by_time["cumulative"], int(total * DEFAULT_FRACTIONS["valid"]))
    calibration_end = _boundary_time(counts_by_time["evaluation_time"], counts_by_time["cumulative"], int(total * DEFAULT_FRACTIONS["calibration"]))
    embargo_2_end = _boundary_time(counts_by_time["evaluation_time"], counts_by_time["cumulative"], int(total * DEFAULT_FRACTIONS["embargo_2"]))

    def split_name(ts: pd.Timestamp) -> str:
        if ts <= train_end:
            return "train"
        if ts <= embargo_1_end:
            return "embargo"
        if ts <= valid_end:
            return "valid"
        if ts <= calibration_end:
            return "calibration"
        if ts <= embargo_2_end:
            return "embargo"
        return "test"

    eval_points["split"] = eval_points["evaluation_time"].map(split_name)
    eval_points.to_parquet(eval_path, index=False)

    boundary_payload = {
        "train_end": train_end.isoformat(),
        "embargo_1_end": embargo_1_end.isoformat(),
        "valid_end": valid_end.isoformat(),
        "calibration_end": calibration_end.isoformat(),
        "embargo_2_end": embargo_2_end.isoformat(),
        "fractions": DEFAULT_FRACTIONS,
        "num_eval_points": total,
    }
    write_json(output_root / "eval_points" / "split_boundaries.json", boundary_payload)

    summary_rows = []
    for split, split_df in eval_points.groupby("split", sort=False):
        summary_rows.append(
            {
                "split": split,
                "num_eval_points": int(len(split_df)),
                "num_unique_entities": int(split_df["entity_id"].nunique()),
                "num_positive_labels": int(pd.to_numeric(split_df["label"], errors="coerce").fillna(0).sum()),
                "min_evaluation_time": str(split_df["evaluation_time"].min()),
                "max_evaluation_time": str(split_df["evaluation_time"].max()),
            }
        )
    write_json(output_root / "split_summary_pre_encode.json", {"splits": summary_rows})
    return eval_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    args = parser.parse_args()
    assign_splits(args.output_root)


if __name__ == "__main__":
    main()
