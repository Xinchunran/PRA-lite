from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.inference.extract_embeddings import REPRESENTATION_TYPES, extract_embeddings_dataframe, load_extractor_artifacts
from src.pragma_c.common import build_account_event_view, compute_profile_state, history_before
from src.tokenizer.structured import StructuredRecordConfig, encode_record


NATURE_BLUE = "#457b9d"
NATURE_TEAL = "#2a9d8f"
NATURE_CYAN = "#76c7c0"
NATURE_NAVY = "#1f5c7a"
NATURE_CORAL = "#e76f51"
MODEL_COLORS = {
    "pragma_lite_logreg": NATURE_BLUE,
    "xgboost": NATURE_TEAL,
    "catboost": NATURE_CYAN,
}


def _safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _safe_json(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float = 1.0) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = sorted({0.5, *y_score.tolist()})
    best_threshold = 0.5
    best_value = -1.0
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(np.int64)
        value = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if value > best_value:
            best_threshold = float(threshold)
            best_value = float(value)
    return best_threshold


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
    }


def _setup_plotting() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
        }
    )


def _balanced_sample_eval_points(
    eval_points: pd.DataFrame,
    sample_size: int,
    seed: int,
    positive_fraction: float,
) -> pd.DataFrame:
    eligible = eval_points[eval_points["split"].isin(["train", "valid", "calibration", "test"])].copy()
    if sample_size >= len(eligible):
        return eligible.reset_index(drop=True)
    split_counts = eligible["split"].value_counts()
    split_order = ["train", "valid", "calibration", "test"]
    split_budgets: dict[str, int] = {}
    remaining = sample_size
    total = int(len(eligible))
    for split_name in split_order[:-1]:
        count = int(split_counts.get(split_name, 0))
        budget = min(count, max(1, int(round(sample_size * count / max(total, 1)))))
        split_budgets[split_name] = budget
        remaining -= budget
    split_budgets[split_order[-1]] = min(int(split_counts.get(split_order[-1], 0)), max(1, remaining))

    sampled_parts: list[pd.DataFrame] = []
    for offset, split_name in enumerate(split_order):
        split_frame = eligible[eligible["split"] == split_name]
        if split_frame.empty:
            continue
        budget = min(len(split_frame), split_budgets.get(split_name, 0))
        pos_frame = split_frame[split_frame["label"] == 1]
        neg_frame = split_frame[split_frame["label"] == 0]
        target_pos = min(len(pos_frame), int(round(budget * positive_fraction)))
        if len(pos_frame) > 0 and target_pos == 0:
            target_pos = 1
        target_neg = budget - target_pos
        if target_neg > len(neg_frame):
            target_neg = len(neg_frame)
            target_pos = min(len(pos_frame), budget - target_neg)
        sampled_split = []
        if target_pos > 0:
            sampled_split.append(pos_frame.sample(n=target_pos, random_state=seed + offset, replace=False))
        if target_neg > 0:
            sampled_split.append(neg_frame.sample(n=target_neg, random_state=seed + 101 + offset, replace=False))
        if sampled_split:
            sampled_parts.append(pd.concat(sampled_split, ignore_index=False))
    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) < sample_size:
        extra = eligible.loc[~eligible.index.isin(sampled.index)]
        take = min(len(extra), sample_size - len(sampled))
        if take > 0:
            pos_extra = extra[extra["label"] == 1]
            neg_extra = extra[extra["label"] == 0]
            want_pos = min(len(pos_extra), int(round(take * positive_fraction)))
            want_neg = min(len(neg_extra), take - want_pos)
            fill_parts = []
            if want_pos > 0:
                fill_parts.append(pos_extra.sample(n=want_pos, random_state=seed + 997, replace=False))
            if want_neg > 0:
                fill_parts.append(neg_extra.sample(n=want_neg, random_state=seed + 1997, replace=False))
            sampled = pd.concat([sampled, *fill_parts], ignore_index=False)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed, replace=False)
    return sampled.sort_values(["split", "evaluation_time", "entity_id", "anchor_transaction_id"], kind="stable").reset_index(drop=True)


def _load_eval_points(stream_root: Path) -> pd.DataFrame:
    eval_path = stream_root / "eval_points" / "eval_points.parquet"
    eval_columns = [
        "eval_id",
        "entity_id",
        "evaluation_time",
        "anchor_transaction_id",
        "label",
        "split",
        "eval_source",
    ]
    eval_points = (
        ds.dataset(eval_path)
        .to_table(columns=eval_columns, filter=ds.field("task") == "downstream")
        .to_pandas()
    )
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    return eval_points


def _load_transactions_for_entities(stream_root: Path, entity_ids: list[int]) -> pd.DataFrame:
    if not entity_ids:
        return pd.DataFrame()
    tx_path = stream_root / "canonical" / "transactions.parquet"
    tx_columns = [
        "transaction_id",
        "transaction_time",
        "from_bank",
        "to_bank",
        "sender_entity_id",
        "receiver_entity_id",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    filter_expr = ds.field("sender_entity_id").isin(entity_ids) | ds.field("receiver_entity_id").isin(entity_ids)
    tx = ds.dataset(tx_path).to_table(columns=tx_columns, filter=filter_expr).to_pandas()
    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")
    tx = tx.dropna(subset=["transaction_time"]).sort_values("transaction_time", kind="stable").reset_index(drop=True)
    return tx


def _build_anchor_feature_row(anchor_row: dict[str, Any] | None) -> dict[str, Any]:
    if anchor_row is None:
        return {
            "anchor_amount_paid": 0.0,
            "anchor_amount_received": 0.0,
            "anchor_payment_format": "UNK",
            "anchor_payment_currency": "UNK",
            "anchor_receiving_currency": "UNK",
            "anchor_from_bank": "UNK",
            "anchor_to_bank": "UNK",
            "anchor_same_bank": 0,
        }
    from_bank = str(anchor_row.get("from_bank", "UNK") or "UNK")
    to_bank = str(anchor_row.get("to_bank", "UNK") or "UNK")
    return {
        "anchor_amount_paid": float(anchor_row.get("amount_paid", 0.0) or 0.0),
        "anchor_amount_received": float(anchor_row.get("amount_received", 0.0) or 0.0),
        "anchor_payment_format": str(anchor_row.get("payment_format", "UNK") or "UNK"),
        "anchor_payment_currency": str(anchor_row.get("payment_currency", "UNK") or "UNK"),
        "anchor_receiving_currency": str(anchor_row.get("receiving_currency", "UNK") or "UNK"),
        "anchor_from_bank": from_bank,
        "anchor_to_bank": to_bank,
        "anchor_same_bank": int(from_bank == to_bank),
    }


def _build_benchmark_dataset(
    stream_root: Path,
    sampled_eval: pd.DataFrame,
    checkpoint: Path,
    output_dir: Path,
    max_history_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_extractor_artifacts(checkpoint, device="cpu")
    vocab = artifacts.vocab
    cfg = StructuredRecordConfig(
        max_events=int(artifacts.checkpoint["model_cfg"]["max_events"]),
        max_event_tokens=int(artifacts.checkpoint["model_cfg"]["max_event_tokens"]),
        max_profile_tokens=int(artifacts.checkpoint["model_cfg"]["max_profile_tokens"]),
        history_time_anchor="last_event",
    )

    involved_entities = sorted({int(x) for x in sampled_eval["entity_id"].unique().tolist()})
    tx = _load_transactions_for_entities(stream_root, involved_entities)
    account_events = build_account_event_view(tx)
    event_groups = {int(entity_id): df.reset_index(drop=True) for entity_id, df in account_events.groupby("entity_id", sort=False)}
    anchor_lookup = {
        int(row["transaction_id"]): row
        for row in tx.to_dict(orient="records")
    }

    tokenized_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    ordered_eval = sampled_eval.sort_values(["split", "evaluation_time", "entity_id", "anchor_transaction_id"], kind="stable")
    for sample_id, row in enumerate(ordered_eval.itertuples(index=False), start=1):
        entity_id = int(row.entity_id)
        entity_events = event_groups.get(entity_id)
        if entity_events is None:
            history = account_events.iloc[0:0].copy()
        else:
            history = history_before(entity_events, row.evaluation_time, max_history_events=max_history_events)
        profile = compute_profile_state(history, row.evaluation_time)
        encoded = encode_record(
            vocab=vocab,
            profile=profile,
            events=history,
            evaluation_time=row.evaluation_time,
            cfg=cfg,
        )
        tokenized_rows.append(
            {
                "entity_id": int(sample_id),
                **encoded,
                "label": int(row.label),
                "batching_event_count": int(sum(encoded["event_mask"])),
                "batching_profile_token_count": int(sum(encoded["profile_mask"])),
            }
        )
        anchor_features = _build_anchor_feature_row(anchor_lookup.get(int(row.anchor_transaction_id)))
        raw_rows.append(
            {
                "entity_id": int(sample_id),
                "label": int(row.label),
                "split": str(row.split),
                "eval_source": str(row.eval_source),
                "history_event_count": int(len(history)),
                **profile,
                **anchor_features,
            }
        )
        metadata_rows.append(
            {
                "entity_id": int(sample_id),
                "orig_entity_id": entity_id,
                "eval_id": str(row.eval_id),
                "split": str(row.split),
                "label": int(row.label),
                "evaluation_time": row.evaluation_time.isoformat(),
                "anchor_transaction_id": int(row.anchor_transaction_id),
                "eval_source": str(row.eval_source),
            }
        )

    benchmark_data_dir = output_dir / "benchmark_data"
    tokenized_dir = benchmark_data_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows).sort_values("entity_id").reset_index(drop=True)
    tokenized_df = pd.DataFrame(tokenized_rows).sort_values("entity_id").reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata_rows).sort_values("entity_id").reset_index(drop=True)
    tokenized_df.to_parquet(tokenized_dir / "dataset.parquet", index=False)
    raw_df.to_parquet(benchmark_data_dir / "raw_features.parquet", index=False)
    metadata_df.to_parquet(benchmark_data_dir / "sample_metadata.parquet", index=False)
    ordered_eval.to_parquet(benchmark_data_dir / "sampled_eval_points.parquet", index=False)
    return raw_df.merge(metadata_df, on=["entity_id", "split", "label"], how="left"), tokenized_df


def _prepare_tabular_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for col in x.columns:
        if col in {"entity_id", "label", "split", "eval_id", "evaluation_time"}:
            continue
        if pd.api.types.is_bool_dtype(x[col]):
            x[col] = x[col].astype(np.int8)
        elif pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].astype(np.float32).fillna(0.0)
        else:
            x[col] = x[col].astype("string").fillna("missing")
    return x


def _train_test_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    test_mask = frame["split"] == "test"
    dev_mask = frame["split"].isin(["train", "valid", "calibration"])
    return dev_mask, test_mask


def _build_cv_predictions_logreg(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    params: dict[str, Any],
    seed: int,
    cv_folds: int,
) -> np.ndarray:
    oof = np.zeros(len(y_dev), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x_dev, y_dev):
        scaler = StandardScaler()
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed, **params)
        x_train = scaler.fit_transform(x_dev[train_idx])
        x_valid = scaler.transform(x_dev[valid_idx])
        clf.fit(x_train, y_dev[train_idx])
        oof[valid_idx] = clf.predict_proba(x_valid)[:, 1]
    return oof


def run_pragma_lite_cv(
    embeddings_df: pd.DataFrame,
    metrics_dir: Path,
    predictions_dir: Path,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_cols = [col for col in embeddings_df.columns if col.startswith("embedding_")]
    dev_mask, test_mask = _train_test_masks(embeddings_df)
    dev_df = embeddings_df.loc[dev_mask].reset_index(drop=True)
    test_df = embeddings_df.loc[test_mask].reset_index(drop=True)
    x_dev = dev_df[feature_cols].to_numpy(dtype=np.float32)
    y_dev = dev_df["label"].to_numpy(dtype=np.int64)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    cv_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_metric = -1.0
    best_oof: np.ndarray | None = None
    for params in (
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ):
        oof = _build_cv_predictions_logreg(x_dev, y_dev, params=params, seed=seed, cv_folds=cv_folds)
        pr_auc = float(average_precision_score(y_dev, oof))
        roc_auc = float(roc_auc_score(y_dev, oof))
        threshold = _best_threshold(y_dev, oof, beta=1.0)
        cv_row = {
            "model_name": "pragma_lite_logreg",
            "params": params,
            "cv_pr_auc": pr_auc,
            "cv_roc_auc": roc_auc,
            "cv_threshold": threshold,
        }
        cv_rows.append(cv_row)
        if pr_auc > best_metric:
            best_metric = pr_auc
            best_params = params
            best_oof = oof
    assert best_params is not None and best_oof is not None
    threshold = _best_threshold(y_dev, best_oof, beta=1.0)
    scaler = StandardScaler()
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed, **best_params)
    clf.fit(scaler.fit_transform(x_dev), y_dev)
    test_score = clf.predict_proba(scaler.transform(x_test))[:, 1]
    pd.DataFrame(
        {
            "entity_id": test_df["entity_id"].astype(np.int64),
            "label": y_test.astype(np.int64),
            "probability": test_score.astype(np.float64),
        }
    ).to_parquet(predictions_dir / "pragma_lite_logreg_test_predictions.parquet", index=False)
    result = {
        "model_name": "pragma_lite_logreg",
        "best_params": best_params,
        "cv_metrics": compute_binary_metrics(y_dev, best_oof, threshold=threshold),
        "test_metrics": compute_binary_metrics(y_test, test_score, threshold=threshold),
    }
    (metrics_dir / "pragma_lite_logreg_cv.json").write_text(json.dumps(_safe_json(cv_rows), indent=2) + "\n", encoding="utf-8")
    return result, cv_rows


def _prepare_xgb_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = _prepare_tabular_frame(frame)
    feature_cols = [col for col in x.columns if col not in {"entity_id", "label", "split", "eval_id", "evaluation_time"}]
    x = x[feature_cols].copy()
    cat_cols = [col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])]
    return pd.get_dummies(x, columns=cat_cols, dummy_na=False)


def run_xgboost_cv(
    raw_df: pd.DataFrame,
    metrics_dir: Path,
    predictions_dir: Path,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required for the IBM AML downstream benchmark") from exc
    dev_mask, test_mask = _train_test_masks(raw_df)
    prepared = _prepare_xgb_frame(raw_df)
    dev_x = prepared.loc[dev_mask].reset_index(drop=True)
    test_x = prepared.loc[test_mask].reset_index(drop=True)
    dev_y = raw_df.loc[dev_mask, "label"].to_numpy(dtype=np.int64)
    test_y = raw_df.loc[test_mask, "label"].to_numpy(dtype=np.int64)

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_metric = -1.0
    best_oof: np.ndarray | None = None
    for params in (
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0},
    ):
        oof = np.zeros(len(dev_y), dtype=np.float64)
        for train_idx, valid_idx in splitter.split(dev_x, dev_y):
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                tree_method="hist",
                n_jobs=1,
                **params,
            )
            clf.fit(dev_x.iloc[train_idx], dev_y[train_idx])
            oof[valid_idx] = clf.predict_proba(dev_x.iloc[valid_idx])[:, 1]
        pr_auc = float(average_precision_score(dev_y, oof))
        roc_auc = float(roc_auc_score(dev_y, oof))
        threshold = _best_threshold(dev_y, oof, beta=1.0)
        cv_row = {
            "model_name": "xgboost",
            "params": params,
            "cv_pr_auc": pr_auc,
            "cv_roc_auc": roc_auc,
            "cv_threshold": threshold,
        }
        cv_rows.append(cv_row)
        if pr_auc > best_metric:
            best_metric = pr_auc
            best_params = params
            best_oof = oof
    assert best_params is not None and best_oof is not None
    threshold = _best_threshold(dev_y, best_oof, beta=1.0)
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=1,
        **best_params,
    )
    clf.fit(dev_x, dev_y)
    test_score = clf.predict_proba(test_x)[:, 1]
    pd.DataFrame(
        {
            "entity_id": raw_df.loc[test_mask, "entity_id"].astype(np.int64),
            "label": test_y.astype(np.int64),
            "probability": test_score.astype(np.float64),
        }
    ).to_parquet(predictions_dir / "xgboost_test_predictions.parquet", index=False)
    result = {
        "model_name": "xgboost",
        "best_params": best_params,
        "cv_metrics": compute_binary_metrics(dev_y, best_oof, threshold=threshold),
        "test_metrics": compute_binary_metrics(test_y, test_score, threshold=threshold),
    }
    (metrics_dir / "xgboost_cv.json").write_text(json.dumps(_safe_json(cv_rows), indent=2) + "\n", encoding="utf-8")
    return result, cv_rows


def _prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    x = _prepare_tabular_frame(frame)
    feature_cols = [col for col in x.columns if col not in {"entity_id", "label", "split", "eval_id", "evaluation_time"}]
    x = x[feature_cols].copy()
    cat_features = [idx for idx, col in enumerate(x.columns) if not pd.api.types.is_numeric_dtype(x[col])]
    return x, cat_features


def run_catboost_cv(
    raw_df: pd.DataFrame,
    metrics_dir: Path,
    predictions_dir: Path,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("catboost is required for the IBM AML downstream benchmark") from exc
    dev_mask, test_mask = _train_test_masks(raw_df)
    prepared, cat_features = _prepare_catboost_frame(raw_df)
    dev_x = prepared.loc[dev_mask].reset_index(drop=True)
    test_x = prepared.loc[test_mask].reset_index(drop=True)
    dev_y = raw_df.loc[dev_mask, "label"].to_numpy(dtype=np.int64)
    test_y = raw_df.loc[test_mask, "label"].to_numpy(dtype=np.int64)

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_metric = -1.0
    best_oof: np.ndarray | None = None
    for params in (
        {"iterations": 200, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 300, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 5.0},
        {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
    ):
        oof = np.zeros(len(dev_y), dtype=np.float64)
        for train_idx, valid_idx in splitter.split(dev_x, dev_y):
            clf = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="PRAUC",
                random_seed=seed,
                verbose=False,
                **params,
            )
            clf.fit(dev_x.iloc[train_idx], dev_y[train_idx], cat_features=cat_features)
            oof[valid_idx] = clf.predict_proba(dev_x.iloc[valid_idx])[:, 1]
        pr_auc = float(average_precision_score(dev_y, oof))
        roc_auc = float(roc_auc_score(dev_y, oof))
        threshold = _best_threshold(dev_y, oof, beta=1.0)
        cv_row = {
            "model_name": "catboost",
            "params": params,
            "cv_pr_auc": pr_auc,
            "cv_roc_auc": roc_auc,
            "cv_threshold": threshold,
        }
        cv_rows.append(cv_row)
        if pr_auc > best_metric:
            best_metric = pr_auc
            best_params = params
            best_oof = oof
    assert best_params is not None and best_oof is not None
    threshold = _best_threshold(dev_y, best_oof, beta=1.0)
    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=seed,
        verbose=False,
        **best_params,
    )
    clf.fit(dev_x, dev_y, cat_features=cat_features)
    test_score = clf.predict_proba(test_x)[:, 1]
    pd.DataFrame(
        {
            "entity_id": raw_df.loc[test_mask, "entity_id"].astype(np.int64),
            "label": test_y.astype(np.int64),
            "probability": test_score.astype(np.float64),
        }
    ).to_parquet(predictions_dir / "catboost_test_predictions.parquet", index=False)
    result = {
        "model_name": "catboost",
        "best_params": best_params,
        "cv_metrics": compute_binary_metrics(dev_y, best_oof, threshold=threshold),
        "test_metrics": compute_binary_metrics(test_y, test_score, threshold=threshold),
    }
    (metrics_dir / "catboost_cv.json").write_text(json.dumps(_safe_json(cv_rows), indent=2) + "\n", encoding="utf-8")
    return result, cv_rows


def _plot_test_metrics(report: dict[str, Any], plots_dir: Path) -> None:
    _setup_plotting()
    metrics = ["pr_auc", "roc_auc", "f1", "f0_5"]
    model_names = list(report["results"].keys())
    if not model_names:
        return
    x = np.arange(len(metrics))
    width = min(0.24, 0.8 / max(len(model_names), 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for idx, model_name in enumerate(model_names):
        values = [float(report["results"][model_name]["test_metrics"][metric] or 0.0) for metric in metrics]
        offset = (idx - (len(model_names) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            color=MODEL_COLORS.get(model_name, NATURE_BLUE),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["PR-AUC", "ROC-AUC", "F1", "F0.5"])
    ax.set_ylabel("Test Score")
    ax.set_title("IBM AML Downstream Benchmark On Test Set")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "test_metric_bars.png", bbox_inches="tight")
    plt.close(fig)


def _run_model_suite(
    raw_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    metrics_dir: Path,
    predictions_dir: Path,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    results: dict[str, Any] = {}
    cv_rows: dict[str, list[dict[str, Any]]] = {}
    skipped_models: dict[str, dict[str, str]] = {}
    model_specs = (
        (
            "pragma_lite_logreg",
            lambda: run_pragma_lite_cv(
                embeddings_df=embeddings_df,
                metrics_dir=metrics_dir,
                predictions_dir=predictions_dir,
                seed=seed,
                cv_folds=cv_folds,
            ),
            False,
        ),
        (
            "xgboost",
            lambda: run_xgboost_cv(
                raw_df=raw_df,
                metrics_dir=metrics_dir,
                predictions_dir=predictions_dir,
                seed=seed,
                cv_folds=cv_folds,
            ),
            True,
        ),
        (
            "catboost",
            lambda: run_catboost_cv(
                raw_df=raw_df,
                metrics_dir=metrics_dir,
                predictions_dir=predictions_dir,
                seed=seed,
                cv_folds=cv_folds,
            ),
            True,
        ),
    )
    for model_name, runner, optional in model_specs:
        print(f"[ibm_downstream] run {model_name} cv", flush=True)
        try:
            result, model_cv_rows = runner()
        except ImportError as exc:
            if not optional:
                raise
            reason = str(exc)
            skipped_models[model_name] = {
                "status": "skipped_missing_dependency",
                "reason": reason,
            }
            print(f"[ibm_downstream] skip {model_name}: {reason}", flush=True)
            continue
        results[model_name] = result
        cv_rows[model_name] = model_cv_rows
    return results, cv_rows, skipped_models


def run_ibm_aml_downstream_benchmark(
    checkpoint: Path,
    stream_root: Path,
    output_dir: Path,
    sample_size: int,
    batch_size: int,
    device: str,
    seed: int,
    repr_type: str,
    cv_folds: int,
    max_history_events: int,
    positive_fraction: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ibm_downstream] load_eval_points stream_root={stream_root}", flush=True)
    eval_points = _load_eval_points(stream_root)
    print(f"[ibm_downstream] total_downstream_eval_points={len(eval_points)}", flush=True)
    sampled_eval = _balanced_sample_eval_points(
        eval_points,
        sample_size=sample_size,
        seed=seed,
        positive_fraction=positive_fraction,
    )
    print(
        f"[ibm_downstream] sampled_eval_points={len(sampled_eval)} split_counts={sampled_eval['split'].value_counts().to_dict()}",
        flush=True,
    )
    print("[ibm_downstream] build_benchmark_dataset", flush=True)
    raw_df, _ = _build_benchmark_dataset(
        stream_root=stream_root,
        sampled_eval=sampled_eval,
        checkpoint=checkpoint,
        output_dir=output_dir,
        max_history_events=max_history_events,
    )
    print(f"[ibm_downstream] raw_feature_rows={len(raw_df)}", flush=True)
    print(f"[ibm_downstream] extract_embeddings repr_type={repr_type}", flush=True)
    embeddings_df = extract_embeddings_dataframe(
        checkpoint=checkpoint,
        data_dir=output_dir / "benchmark_data" / "tokenized",
        split="all",
        split_dir=None,
        batch_size=batch_size,
        device=device,
        repr_type=repr_type,
    )
    metadata = pd.read_parquet(output_dir / "benchmark_data" / "sample_metadata.parquet")
    embeddings_df = embeddings_df.merge(metadata, on=["entity_id", "label"], how="left")
    embeddings_df.to_parquet(output_dir / "benchmark_data" / f"pragma_lite_{repr_type}_embeddings.parquet", index=False)

    results, cv_rows, skipped_models = _run_model_suite(
        raw_df=raw_df,
        embeddings_df=embeddings_df,
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
        seed=seed,
        cv_folds=cv_folds,
    )

    report = {
        "task": "ibm_aml_downstream_benchmark",
        "checkpoint": str(checkpoint),
        "stream_root": str(stream_root),
        "output_dir": str(output_dir),
        "sample_size": int(len(sampled_eval)),
        "requested_sample_size": int(sample_size),
        "positive_fraction": float(positive_fraction),
        "repr_type": repr_type,
        "seed": seed,
        "cv_folds": cv_folds,
        "results": results,
        "cv_rows": cv_rows,
        "skipped_models": skipped_models,
        "split_counts": sampled_eval["split"].value_counts().sort_index().to_dict(),
        "label_counts": sampled_eval["label"].value_counts().sort_index().to_dict(),
    }
    (metrics_dir / "benchmark_report.json").write_text(json.dumps(_safe_json(report), indent=2) + "\n", encoding="utf-8")
    _plot_test_metrics(report, plots_dir=plots_dir)
    print(f"[ibm_downstream] done output_dir={output_dir}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stream_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repr_type", default="concat", choices=REPRESENTATION_TYPES)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--max_history_events", type=int, default=6500)
    parser.add_argument("--positive_fraction", type=float, default=0.1)
    args = parser.parse_args()
    run_ibm_aml_downstream_benchmark(
        checkpoint=Path(args.checkpoint),
        stream_root=Path(args.stream_root),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        repr_type=args.repr_type,
        cv_folds=args.cv_folds,
        max_history_events=args.max_history_events,
        positive_fraction=args.positive_fraction,
    )


if __name__ == "__main__":
    main()
