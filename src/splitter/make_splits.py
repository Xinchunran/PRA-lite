from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir


def _write_ids(path: Path, ids: np.ndarray) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(str(int(x)) for x in ids) + "\n", encoding="utf-8")


def _read_labels(processed_dir: Path) -> pd.DataFrame:
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    if "label" not in labels.columns:
        raise ValueError("labels.parquet must contain 'label' column")
    return labels


def split_entity(labels: pd.DataFrame, train_size: float, valid_size: float, test_size: float, seed: int) -> dict:
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    rng = np.random.default_rng(seed)
    ids = labels["entity_id"].to_numpy().copy()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_size))
    n_valid = int(round(n * valid_size))
    n_test = n - n_train - n_valid

    train_ids = ids[:n_train]
    valid_ids = ids[n_train : n_train + n_valid]
    test_ids = ids[n_train + n_valid :]

    return {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "summary": {"n_total": n, "n_train": len(train_ids), "n_valid": len(valid_ids), "n_test": len(test_ids)},
    }


def split_time(labels: pd.DataFrame, train_end: str, valid_end: str, test_end: str) -> dict:
    t_train = pd.Timestamp(train_end, tz="UTC")
    t_valid = pd.Timestamp(valid_end, tz="UTC")
    t_test = pd.Timestamp(test_end, tz="UTC")

    eval_time = pd.to_datetime(labels["evaluation_time"], utc=True, errors="coerce")
    if eval_time.isna().any():
        raise ValueError("labels.parquet has invalid evaluation_time values")

    train_ids = labels.loc[eval_time <= t_train, "entity_id"].to_numpy()
    valid_ids = labels.loc[(eval_time > t_train) & (eval_time <= t_valid), "entity_id"].to_numpy()
    test_ids = labels.loc[(eval_time > t_valid) & (eval_time <= t_test), "entity_id"].to_numpy()

    return {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "summary": {
            "n_total": int(len(labels)),
            "n_train": int(len(train_ids)),
            "n_valid": int(len(valid_ids)),
            "n_test": int(len(test_ids)),
            "train_end": train_end,
            "valid_end": valid_end,
            "test_end": test_end,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_mode", required=True, choices=["entity", "time", "random"])
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--train_end")
    parser.add_argument("--valid_end")
    parser.add_argument("--test_end")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    labels = _read_labels(processed_dir)
    if args.split_mode in {"entity", "random"}:
        result = split_entity(labels, args.train_size, args.valid_size, args.test_size, args.seed)
    else:
        if not (args.train_end and args.valid_end and args.test_end):
            raise ValueError("time split requires --train_end --valid_end --test_end")
        result = split_time(labels, args.train_end, args.valid_end, args.test_end)

    _write_ids(out_dir / "train_ids.txt", result["train_ids"])
    _write_ids(out_dir / "valid_ids.txt", result["valid_ids"])
    _write_ids(out_dir / "test_ids.txt", result["test_ids"])

    (out_dir / "split_summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
