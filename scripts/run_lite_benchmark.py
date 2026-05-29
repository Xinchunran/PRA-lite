from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.training.linear_probe import run_probe_experiment


def _load_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{entity_id}\n" for entity_id in ids), encoding="utf-8")


def _sample_split_ids(split_dir: Path, output_dir: Path, seed: int, num_records: int) -> Path:
    rng = np.random.default_rng(seed)
    ratio = {"train": 0.70, "valid": 0.15, "test": 0.15}
    sampled_dir = output_dir / "sampled_splits"
    for split_name, frac in ratio.items():
        source_ids = _load_ids(split_dir / f"{split_name}_ids.txt")
        target_n = min(len(source_ids), max(1, int(round(num_records * frac))))
        sampled = rng.choice(source_ids, size=target_n, replace=False).tolist() if target_n < len(source_ids) else source_ids
        _write_ids(sampled_dir / f"{split_name}_ids.txt", sorted(int(x) for x in sampled))
    return sampled_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_records", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--repr_types",
        nargs="+",
        default=["zh_usr", "last_evt", "concat", "record"],
        choices=["zh_usr", "last_evt", "concat", "record"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_split_dir = _sample_split_ids(
        split_dir=Path(args.split_dir),
        output_dir=output_dir,
        seed=args.seed,
        num_records=args.num_records,
    )

    aggregate: dict[str, object] = {
        "seed": args.seed,
        "num_records": args.num_records,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split_dir": str(sampled_split_dir),
        "repr_types": args.repr_types,
        "results": {},
    }
    for repr_type in args.repr_types:
        probe_dir = output_dir / repr_type
        report = run_probe_experiment(
            checkpoint=Path(args.checkpoint),
            data_dir=Path(args.data_dir),
            split_dir=sampled_split_dir,
            output_dir=probe_dir,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            repr_type=repr_type,
        )
        aggregate["results"][repr_type] = {
            "valid_pr_auc": report.get("valid_metrics", {}).get("pr_auc"),
            "valid_roc_auc": report.get("valid_metrics", {}).get("roc_auc"),
            "test_pr_auc": report.get("test_metrics", {}).get("pr_auc"),
            "test_roc_auc": report.get("test_metrics", {}).get("roc_auc"),
        }

    (output_dir / "benchmark_report.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
