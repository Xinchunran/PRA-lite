from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.inference.extract_embeddings import (
    REPRESENTATION_TYPES,
    build_tokenized_loader,
    collect_representations,
    load_extractor_artifacts,
)
from src.training.data import load_tokenized_split, set_seed


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int | None]:
    y_pred = (y_score >= 0.5).astype(np.int64)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "accuracy": float((y_pred == y_true).mean()),
    }


def run_probe_experiment(
    checkpoint: Path,
    data_dir: Path,
    split_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int,
    repr_type: str,
    train_split_name: str = "train",
    eval_splits: tuple[str, ...] = ("valid", "test"),
) -> dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_extractor_artifacts(checkpoint, device=device)

    train_ds = load_tokenized_split(data_dir, train_split_name, split_dir=split_dir)
    train_loader = build_tokenized_loader(train_ds, pad_id=artifacts.vocab.pad_id, batch_size=batch_size)
    x_train, y_train, train_entity_ids = collect_representations(
        artifacts.model,
        train_loader,
        device=device,
        repr_type=repr_type,
        require_labels=True,
    )
    if y_train is None:
        raise ValueError("Tokenized dataset must include label for linear_probe")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(x_train_scaled, y_train)

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "repr_type": repr_type,
        "seed": seed,
        "train": {"n": int(len(y_train)), "pos": int(y_train.sum())},
    }

    train_pred = clf.predict_proba(x_train_scaled)[:, 1]
    report["train_metrics"] = _binary_metrics(y_train, train_pred)

    train_pred_df = pd.DataFrame(
        {
            "entity_id": train_entity_ids.astype(np.int64),
            "label": y_train.astype(np.int64),
            "probability": train_pred.astype(np.float64),
        }
    )
    train_pred_df.to_parquet(output_dir / f"{repr_type}_train_predictions.parquet", index=False)

    for split_name in eval_splits:
        split_ds = load_tokenized_split(data_dir, split_name, split_dir=split_dir)
        split_loader = build_tokenized_loader(split_ds, pad_id=artifacts.vocab.pad_id, batch_size=batch_size)
        x_eval, y_eval, eval_entity_ids = collect_representations(
            artifacts.model,
            split_loader,
            device=device,
            repr_type=repr_type,
            require_labels=True,
        )
        if y_eval is None:
            raise ValueError("Tokenized dataset must include label for linear_probe")
        y_score = clf.predict_proba(scaler.transform(x_eval))[:, 1]
        report[f"{split_name}_metrics"] = _binary_metrics(y_eval, y_score)
        pd.DataFrame(
            {
                "entity_id": eval_entity_ids.astype(np.int64),
                "label": y_eval.astype(np.int64),
                "probability": y_score.astype(np.float64),
            }
        ).to_parquet(output_dir / f"{repr_type}_{split_name}_predictions.parquet", index=False)

    (output_dir / f"{repr_type}_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{repr_type}_probe.pkl").open("wb") as f:
        pickle.dump({"scaler": scaler, "classifier": clf, "repr_type": repr_type}, f)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--repr_type",
        default="concat",
        choices=REPRESENTATION_TYPES,
    )
    args = parser.parse_args()

    run_probe_experiment(
        checkpoint=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        split_dir=Path(args.split_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        repr_type=args.repr_type,
    )


if __name__ == "__main__":
    main()
