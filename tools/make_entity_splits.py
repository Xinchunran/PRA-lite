from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json


def _write_ids(path: Path, ids: np.ndarray) -> None:
    ensure_dir(path.parent)
    values = [str(int(x)) for x in ids.tolist()]
    path.write_text(("\n".join(values) + "\n") if values else "", encoding="utf-8")


def make_entity_splits(
    labels_path: str | Path,
    out_dir: str | Path,
    seed: int = 26,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
) -> Path:
    labels = pd.read_parquet(labels_path)
    if "entity_id" not in labels.columns:
        raise ValueError("labels parquet must contain entity_id")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= valid_frac < 1.0 or train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac and valid_frac must leave room for a non-empty or empty test split")

    entity_ids = labels["entity_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(entity_ids)

    n_total = len(entity_ids)
    n_train = int(n_total * train_frac)
    n_valid = int(n_total * valid_frac)
    train_ids = entity_ids[:n_train]
    valid_ids = entity_ids[n_train : n_train + n_valid]
    test_ids = entity_ids[n_train + n_valid :]

    output = ensure_dir(out_dir)
    pd.DataFrame({"entity_id": train_ids}).to_csv(output / "train.csv", index=False)
    pd.DataFrame({"entity_id": valid_ids}).to_csv(output / "valid.csv", index=False)
    pd.DataFrame({"entity_id": test_ids}).to_csv(output / "test.csv", index=False)

    _write_ids(output / "train_ids.txt", train_ids)
    _write_ids(output / "valid_ids.txt", valid_ids)
    _write_ids(output / "test_ids.txt", test_ids)

    write_json(
        output / "split_summary.json",
        {
            "n_total": int(n_total),
            "n_train": int(len(train_ids)),
            "n_valid": int(len(valid_ids)),
            "n_test": int(len(test_ids)),
            "seed": int(seed),
            "train_frac": float(train_frac),
            "valid_frac": float(valid_frac),
            "test_frac": float(1.0 - train_frac - valid_frac),
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--valid_frac", type=float, default=0.1)
    args = parser.parse_args()
    make_entity_splits(args.labels, args.out_dir, seed=args.seed, train_frac=args.train_frac, valid_frac=args.valid_frac)


if __name__ == "__main__":
    main()
