from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.prediction_file)
    if "label" not in df.columns or df["label"].isna().any():
        raise ValueError("prediction_file must contain non-null 'label' column")
    y_true = df["label"].astype("int64").to_numpy()
    y_score = df["probability"].astype("float64").to_numpy()
    y_pred = (y_score >= 0.5).astype(np.int64)

    out = {
        "n": int(len(df)),
        "pos": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_score)),
    }

    prec, rec, thr = precision_recall_curve(y_true, y_score)
    out["pr_curve_points"] = int(len(thr))
    out["max_f1_over_thresholds"] = float(
        np.max(2 * (prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-12))
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
