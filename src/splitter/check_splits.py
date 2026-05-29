from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_ids(path: Path) -> np.ndarray:
    return np.array([int(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() != ""], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    split_dir = Path(args.split_dir)

    labels = pd.read_parquet(processed_dir / "labels.parquet")
    train_ids = _read_ids(split_dir / "train_ids.txt")
    valid_ids = _read_ids(split_dir / "valid_ids.txt")
    test_ids = _read_ids(split_dir / "test_ids.txt")

    sets = {"train": set(train_ids.tolist()), "valid": set(valid_ids.tolist()), "test": set(test_ids.tolist())}
    leakage = {
        "train_valid_overlap": len(sets["train"].intersection(sets["valid"])),
        "train_test_overlap": len(sets["train"].intersection(sets["test"])),
        "valid_test_overlap": len(sets["valid"].intersection(sets["test"])),
    }

    total_unique = len(set.union(*sets.values()))
    missing = sorted(set(labels["entity_id"].astype("int64").tolist()) - set.union(*sets.values()))

    def _label_stats(ids: np.ndarray) -> dict:
        subset = labels[labels["entity_id"].isin(ids)]
        pos = int(subset["label"].sum())
        n = int(len(subset))
        return {"n": n, "pos": pos, "pos_rate": float(pos / max(n, 1))}

    report = {
        "leakage": leakage,
        "counts": {"train": len(train_ids), "valid": len(valid_ids), "test": len(test_ids), "total_unique": total_unique},
        "missing_entity_ids_count": len(missing),
        "label_stats": {"train": _label_stats(train_ids), "valid": _label_stats(valid_ids), "test": _label_stats(test_ids)},
    }

    out_path = split_dir / "split_check.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if any(v > 0 for v in leakage.values()):
        raise SystemExit(f"Split leakage detected: {leakage}. See {out_path}")
    if len(missing) > 0:
        raise SystemExit(f"Some entity_ids are missing from splits: {len(missing)}. See {out_path}")


if __name__ == "__main__":
    main()
