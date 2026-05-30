#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.fs import read_json, write_json


def build_manifest(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    tokenizer_dir = output_root / "tokenizer"
    split_boundaries_path = output_root / "eval_points" / "split_boundaries.json"
    shard_root = output_root / "tokenized_shards"

    shard_entries = []
    split_summary: dict[str, dict[str, int]] = {}
    for shard_dir in sorted(shard_root.glob("shard_*")):
        summary_path = shard_dir / "shard_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        counts = summary.get("counts", {})
        shard_entries.append(
            {
                "name": shard_dir.name,
                "tokenized_dir": str(shard_dir.resolve()),
                "status": "ready",
                "splits": {split_name: f"{split_name}.lmdb" for split_name in ("train", "valid", "calibration", "test", "embargo")},
                "counts": counts,
            }
        )
        for split_name, split_count in counts.items():
            split_summary.setdefault(split_name, {"num_records": 0})
            split_summary[split_name]["num_records"] += int(split_count)

    manifest = {
        "dataset_name": "ibm_aml_li_medium_pragma_c",
        "dataset_stage": "C",
        "record_policy": "multi_evaluation_point_pragma_style",
        "split_policy": "global_evaluation_time_with_embargo_and_calibration",
        "split_boundaries_path": str(split_boundaries_path.resolve()),
        "tokenizer_policy": "train_only_fit",
        "history_rule": "transaction_time < evaluation_time",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "shards": shard_entries,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "split_summary.json", {"splits": split_summary})
    return output_root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    args = parser.parse_args()
    build_manifest(args.output_root)


if __name__ == "__main__":
    main()
