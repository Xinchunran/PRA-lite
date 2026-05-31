#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import read_json, write_json


def audit_leakage(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    eval_points = pd.read_parquet(output_root / "eval_points" / "eval_points.parquet")
    boundaries = read_json(output_root / "eval_points" / "split_boundaries.json")
    tokenizer_cfg = read_json(output_root / "tokenizer" / "build_config.json")

    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    train_end = pd.Timestamp(boundaries["train_end"])
    valid_end = pd.Timestamp(boundaries["valid_end"])
    calibration_end = pd.Timestamp(boundaries["calibration_end"])
    embargo_2_end = pd.Timestamp(boundaries["embargo_2_end"])

    violations = []
    if tokenizer_cfg.get("fit_split") != "train":
        violations.append("tokenizer_fit_split_is_not_train")
    if not eval_points.loc[eval_points["split"] == "train", "evaluation_time"].le(train_end).all():
        violations.append("train_eval_points_extend_past_train_end")
    if not eval_points.loc[eval_points["split"] == "valid", "evaluation_time"].gt(train_end).all():
        violations.append("valid_eval_points_overlap_train")
    if not eval_points.loc[eval_points["split"] == "valid", "evaluation_time"].le(valid_end).all():
        violations.append("valid_eval_points_extend_past_valid_end")
    if not eval_points.loc[eval_points["split"] == "calibration", "evaluation_time"].gt(valid_end).all():
        violations.append("calibration_eval_points_overlap_valid")
    if not eval_points.loc[eval_points["split"] == "calibration", "evaluation_time"].le(calibration_end).all():
        violations.append("calibration_eval_points_extend_past_calibration_end")
    if not eval_points.loc[eval_points["split"] == "test", "evaluation_time"].gt(embargo_2_end).all():
        violations.append("test_eval_points_overlap_embargo_2")

    payload = {
        "passes": not violations,
        "violations": violations,
        "checks": {
            "tokenizer_fit_split": tokenizer_cfg.get("fit_split"),
            "num_eval_points": int(len(eval_points)),
            "num_train_eval_points": int((eval_points["split"] == "train").sum()),
        },
    }
    write_json(output_root / "leakage_audit.json", payload)
    return output_root / "leakage_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    args = parser.parse_args()
    audit_leakage(args.output_root)


if __name__ == "__main__":
    main()
