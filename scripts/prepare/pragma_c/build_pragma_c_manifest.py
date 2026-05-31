#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.fs import read_json, write_json


def build_manifest(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    tokenizer_dir = output_root / "tokenizer"
    split_boundaries_path = output_root / "eval_points" / "split_boundaries.json"
    encode_index_summary_path = output_root / "eval_points" / "encode_index_summary.json"
    shard_root = output_root / "tokenized_shards"

    expected_shards: dict[str, dict[str, int]] = {}
    if encode_index_summary_path.exists():
        encode_index_summary = read_json(encode_index_summary_path)
        for row in encode_index_summary.get("rows", []):
            if not isinstance(row, dict):
                continue
            shard_name = f"shard_{int(row.get('shard_index', -1)):05d}"
            if shard_name.endswith("-0001"):
                continue
            expected_shards[shard_name] = {
                "num_eval_points": int(row.get("num_eval_points", 0)),
                "num_unique_entities": int(row.get("num_unique_entities", 0)),
            }

    shard_entries_by_name: dict[str, dict[str, object]] = {}
    split_summary: dict[str, dict[str, int]] = {}
    for shard_dir in sorted(shard_root.glob("shard_*")):
        summary_path = shard_dir / "shard_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        counts = summary.get("counts", {})
        shard_entries_by_name[shard_dir.name] = {
            "name": shard_dir.name,
            "tokenized_dir": str(shard_dir.resolve()),
            "status": "ready",
            "splits": {split_name: f"{split_name}.lmdb" for split_name in ("train", "valid", "calibration", "test", "embargo")},
            "counts": counts,
            "num_records": int(summary.get("num_records", sum(int(value) for value in counts.values()))),
            "num_eval_points": int(expected_shards.get(shard_dir.name, {}).get("num_eval_points", 0)),
            "num_unique_entities": int(expected_shards.get(shard_dir.name, {}).get("num_unique_entities", 0)),
        }
        for split_name, split_count in counts.items():
            split_summary.setdefault(split_name, {"num_records": 0})
            split_summary[split_name]["num_records"] += int(split_count)

    for shard_name, shard_meta in expected_shards.items():
        if shard_name in shard_entries_by_name:
            continue
        shard_entries_by_name[shard_name] = {
            "name": shard_name,
            "tokenized_dir": str((shard_root / shard_name).resolve()),
            "status": "pending",
            "splits": {split_name: f"{split_name}.lmdb" for split_name in ("train", "valid", "calibration", "test", "embargo")},
            "counts": {},
            "num_records": 0,
            "num_eval_points": int(shard_meta.get("num_eval_points", 0)),
            "num_unique_entities": int(shard_meta.get("num_unique_entities", 0)),
        }

    manifest = {
        "dataset_name": "ibm_aml_li_medium_pragma_c",
        "dataset_stage": "C",
        "record_policy": "multi_evaluation_point_pragma_style",
        "split_policy": "global_evaluation_time_with_embargo_and_calibration",
        "split_boundaries_path": str(split_boundaries_path.resolve()),
        "tokenizer_policy": "train_only_fit",
        "history_rule": "transaction_time < evaluation_time",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "expected_shards": len(expected_shards) if expected_shards else len(shard_entries_by_name),
        "shards": [shard_entries_by_name[name] for name in sorted(shard_entries_by_name.keys())],
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
