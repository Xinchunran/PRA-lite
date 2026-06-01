from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.inference.extract_embeddings import REPRESENTATION_TYPES, extract_embeddings_dataframe


def _load_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{entity_id}\n" for entity_id in ids), encoding="utf-8")


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


def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(df):
        return df.copy()
    if "label" not in df.columns or df["label"].nunique(dropna=False) <= 1:
        return df.sample(n=sample_size, random_state=seed, replace=False)
    sampled_parts: list[pd.DataFrame] = []
    remaining = sample_size
    groups = list(df.groupby("label", sort=True))
    total = len(df)
    for label, group in groups[:-1]:
        del label
        target = int(round(sample_size * len(group) / total))
        target = min(len(group), max(1, target))
        sampled_parts.append(group.sample(n=target, random_state=seed, replace=False))
        remaining -= target
    last_group = groups[-1][1]
    last_target = min(len(last_group), max(1, remaining))
    sampled_parts.append(last_group.sample(n=last_target, random_state=seed, replace=False))
    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed, replace=False)
    if len(sampled) < sample_size:
        extra_candidates = df.loc[~df.index.isin(sampled.index)]
        extra_n = min(len(extra_candidates), sample_size - len(sampled))
        if extra_n > 0:
            sampled = pd.concat(
                [sampled, extra_candidates.sample(n=extra_n, random_state=seed, replace=False)],
                ignore_index=False,
            )
    return sampled.sort_values("entity_id").reset_index(drop=True)


def sample_split_ids(
    labels_path: Path,
    split_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> dict[str, list[int]]:
    labels = pd.read_parquet(labels_path, columns=["entity_id", "label"])
    sampled_dir = output_dir / "sampled_splits"
    result: dict[str, list[int]] = {}
    split_names = ("train", "valid", "test")
    available_total = 0
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        split_ids = set(_load_ids(split_dir / f"{split_name}_ids.txt"))
        split_frame = labels[labels["entity_id"].isin(split_ids)].copy()
        split_frames[split_name] = split_frame
        available_total += len(split_frame)
    if available_total == 0:
        raise ValueError(f"No overlapping labels found under split_dir={split_dir}")
    allocated = 0
    target_counts: dict[str, int] = {}
    for split_name in split_names[:-1]:
        split_count = len(split_frames[split_name])
        target = int(round(sample_size * split_count / available_total))
        target = min(split_count, max(1, target)) if split_count > 0 else 0
        target_counts[split_name] = target
        allocated += target
    last_name = split_names[-1]
    last_count = len(split_frames[last_name])
    target_counts[last_name] = min(last_count, max(1 if last_count > 0 else 0, sample_size - allocated))
    for split_name in split_names:
        split_frame = split_frames[split_name]
        sampled_frame = _stratified_sample(split_frame, target_counts[split_name], seed=seed)
        sampled_ids = sampled_frame["entity_id"].astype("int64").sort_values().tolist()
        _write_ids(sampled_dir / f"{split_name}_ids.txt", sampled_ids)
        result[split_name] = sampled_ids
    return result


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _build_event_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["entity_id", "event_count"])
    events = events.copy()
    if "timestamp" in events.columns:
        events["_timestamp"] = _coerce_timestamp(events["timestamp"])
        events = events.sort_values(["entity_id", "_timestamp"], kind="stable")
    else:
        events["_timestamp"] = pd.NaT
        events = events.sort_values(["entity_id"], kind="stable")

    reserved = {"entity_id", "event_id", "label", "is_laundering", "evaluation_time", "_timestamp"}
    numeric_cols = [
        col
        for col in events.columns
        if col not in reserved and pd.api.types.is_numeric_dtype(events[col])
    ]
    categorical_cols = [
        col
        for col in events.columns
        if col not in reserved and not pd.api.types.is_numeric_dtype(events[col])
    ]

    grouped = events.groupby("entity_id", sort=False)
    features = grouped.size().rename("event_count").to_frame()
    for col in numeric_cols:
        stats = grouped[col].agg(["mean", "max", "min", "sum", "std", "last"]).add_prefix(f"evt_{col}_")
        features = features.join(stats, how="left")
    for col in categorical_cols:
        as_string = events[col].astype("string").fillna("missing")
        last_values = as_string.groupby(events["entity_id"], sort=False).last().rename(f"evt_{col}_last")
        nunique = as_string.groupby(events["entity_id"], sort=False).nunique().rename(f"evt_{col}_nunique")
        features = features.join(last_values, how="left")
        features = features.join(nunique, how="left")
    if events["_timestamp"].notna().any():
        ts_group = events.groupby("entity_id", sort=False)["_timestamp"]
        last_timestamp = ts_group.max().astype("int64").div(1_000_000_000)
        ts_features = pd.DataFrame(
            {
                "evt_time_span_hours": (ts_group.max() - ts_group.min()).dt.total_seconds().div(3600.0),
                "evt_last_timestamp": last_timestamp,
            }
        )
        features = features.join(ts_features, how="left")
    features.reset_index(inplace=True)
    return features


def _drop_leaky_profile_columns(profiles: pd.DataFrame) -> pd.DataFrame:
    leak_markers = ("label", "is_laundering", "evaluation")
    keep_cols = [col for col in profiles.columns if col == "entity_id" or not any(marker in col.lower() for marker in leak_markers)]
    return profiles[keep_cols].copy()


def build_raw_feature_table(processed_dir: Path, entity_ids: set[int] | None = None) -> pd.DataFrame:
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    labels = labels[["entity_id", "label"]].copy()
    if entity_ids is not None:
        labels = labels[labels["entity_id"].isin(entity_ids)].copy()

    profiles = _read_optional_parquet(processed_dir / "profiles.parquet")
    if not profiles.empty:
        profiles = _drop_leaky_profile_columns(profiles)
        if entity_ids is not None:
            profiles = profiles[profiles["entity_id"].isin(entity_ids)].copy()
        profile_cols = {col: f"profile_{col}" for col in profiles.columns if col != "entity_id"}
        profiles = profiles.rename(columns=profile_cols)

    events = _read_optional_parquet(processed_dir / "events.parquet")
    if not events.empty and entity_ids is not None:
        events = events[events["entity_id"].isin(entity_ids)].copy()
    event_features = _build_event_features(events)

    features = labels.merge(event_features, on="entity_id", how="left")
    if not profiles.empty:
        features = features.merge(profiles, on="entity_id", how="left")
    return features.sort_values("entity_id").reset_index(drop=True)


def _prepare_tabular_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [col for col in train_df.columns if col not in {"entity_id", "label"}]
    prepared: list[pd.DataFrame] = []
    for frame in (train_df, valid_df, test_df):
        current = frame[features].copy()
        for col in current.columns:
            if pd.api.types.is_bool_dtype(current[col]):
                current[col] = current[col].astype(np.int8)
            elif pd.api.types.is_numeric_dtype(current[col]):
                current[col] = current[col].astype(np.float32).fillna(0.0)
            else:
                current[col] = current[col].astype("string").fillna("missing")
        prepared.append(current)
    return prepared[0], prepared[1], prepared[2]


def _prepare_xgb_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_x, valid_x, test_x = _prepare_tabular_frames(train_df, valid_df, test_df)
    combined = pd.concat([train_x, valid_x, test_x], axis=0, ignore_index=True)
    cat_cols = [col for col in combined.columns if not pd.api.types.is_numeric_dtype(combined[col])]
    encoded = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)
    train_end = len(train_x)
    valid_end = train_end + len(valid_x)
    return (
        encoded.iloc[:train_end].reset_index(drop=True),
        encoded.iloc[train_end:valid_end].reset_index(drop=True),
        encoded.iloc[valid_end:].reset_index(drop=True),
    )


def _prepare_catboost_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    train_x, valid_x, test_x = _prepare_tabular_frames(train_df, valid_df, test_df)
    cat_features = [idx for idx, col in enumerate(train_x.columns) if not pd.api.types.is_numeric_dtype(train_x[col])]
    return train_x, valid_x, test_x, cat_features


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
            best_value = float(value)
            best_threshold = float(threshold)
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


def _save_predictions(
    output_dir: Path,
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    y_score: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "entity_id": frame["entity_id"].astype(np.int64),
            "label": frame["label"].astype(np.int64),
            "probability": y_score.astype(np.float64),
        }
    ).to_parquet(output_dir / f"{model_name}_{split_name}_predictions.parquet", index=False)


def run_pragma_lite_logreg(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    feature_cols = [col for col in train_df.columns if col.startswith("embedding_")]
    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_valid = valid_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    best_report: dict[str, object] | None = None
    for params in (
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ):
        scaler = StandardScaler()
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
            **params,
        )
        clf.fit(scaler.fit_transform(x_train), y_train)
        valid_score = clf.predict_proba(scaler.transform(x_valid))[:, 1]
        threshold = _best_threshold(y_valid, valid_score, beta=1.0)
        valid_metrics = compute_binary_metrics(y_valid, valid_score, threshold=threshold)
        candidate = {
            "model_name": "pragma_lite_logreg",
            "best_params": params,
            "selection_metric": float(valid_metrics["pr_auc"] or 0.0),
            "threshold": threshold,
            "valid_metrics": valid_metrics,
            "artifacts": {"scaler": scaler, "classifier": clf},
        }
        if best_report is None or candidate["selection_metric"] > best_report["selection_metric"]:
            best_report = candidate
    assert best_report is not None
    scaler = best_report["artifacts"]["scaler"]
    clf = best_report["artifacts"]["classifier"]
    valid_score = clf.predict_proba(scaler.transform(x_valid))[:, 1]
    test_score = clf.predict_proba(scaler.transform(x_test))[:, 1]
    threshold = float(best_report["threshold"])
    _save_predictions(output_dir, "pragma_lite_logreg", "valid", valid_df, valid_score)
    _save_predictions(output_dir, "pragma_lite_logreg", "test", test_df, test_score)
    return {
        "model_name": "pragma_lite_logreg",
        "best_params": best_report["best_params"],
        "valid_metrics": compute_binary_metrics(y_valid, valid_score, threshold=threshold),
        "test_metrics": compute_binary_metrics(y_test, test_score, threshold=threshold),
    }


def run_xgboost_grid_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("xgboost is required for the downstream baseline; install from requirements.txt") from exc

    x_train, x_valid, x_test = _prepare_xgb_matrices(train_df, valid_df, test_df)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_valid = valid_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    best_report: dict[str, object] | None = None
    for params in (
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0},
    ):
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            n_jobs=1,
            **params,
        )
        clf.fit(x_train, y_train)
        valid_score = clf.predict_proba(x_valid)[:, 1]
        threshold = _best_threshold(y_valid, valid_score, beta=1.0)
        valid_metrics = compute_binary_metrics(y_valid, valid_score, threshold=threshold)
        candidate = {
            "model_name": "xgboost",
            "best_params": params,
            "selection_metric": float(valid_metrics["pr_auc"] or 0.0),
            "threshold": threshold,
            "classifier": clf,
        }
        if best_report is None or candidate["selection_metric"] > best_report["selection_metric"]:
            best_report = candidate
    assert best_report is not None
    clf = best_report["classifier"]
    valid_score = clf.predict_proba(x_valid)[:, 1]
    test_score = clf.predict_proba(x_test)[:, 1]
    threshold = float(best_report["threshold"])
    _save_predictions(output_dir, "xgboost", "valid", valid_df, valid_score)
    _save_predictions(output_dir, "xgboost", "test", test_df, test_score)
    return {
        "model_name": "xgboost",
        "best_params": best_report["best_params"],
        "valid_metrics": compute_binary_metrics(y_valid, valid_score, threshold=threshold),
        "test_metrics": compute_binary_metrics(y_test, test_score, threshold=threshold),
    }


def run_catboost_grid_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("catboost is required for the downstream baseline; install from requirements.txt") from exc

    x_train, x_valid, x_test, cat_features = _prepare_catboost_frames(train_df, valid_df, test_df)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_valid = valid_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    best_report: dict[str, object] | None = None
    for params in (
        {"iterations": 200, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 300, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 5.0},
        {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
    ):
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=seed,
            verbose=False,
            **params,
        )
        clf.fit(x_train, y_train, cat_features=cat_features)
        valid_score = clf.predict_proba(x_valid)[:, 1]
        threshold = _best_threshold(y_valid, valid_score, beta=1.0)
        valid_metrics = compute_binary_metrics(y_valid, valid_score, threshold=threshold)
        candidate = {
            "model_name": "catboost",
            "best_params": params,
            "selection_metric": float(valid_metrics["pr_auc"] or 0.0),
            "threshold": threshold,
            "classifier": clf,
        }
        if best_report is None or candidate["selection_metric"] > best_report["selection_metric"]:
            best_report = candidate
    assert best_report is not None
    clf = best_report["classifier"]
    valid_score = clf.predict_proba(x_valid)[:, 1]
    test_score = clf.predict_proba(x_test)[:, 1]
    threshold = float(best_report["threshold"])
    _save_predictions(output_dir, "catboost", "valid", valid_df, valid_score)
    _save_predictions(output_dir, "catboost", "test", test_df, test_score)
    return {
        "model_name": "catboost",
        "best_params": best_report["best_params"],
        "valid_metrics": compute_binary_metrics(y_valid, valid_score, threshold=threshold),
        "test_metrics": compute_binary_metrics(y_test, test_score, threshold=threshold),
    }


def _extract_split_embeddings(
    checkpoint: Path,
    tokenized_dir: Path,
    sampled_split_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    repr_type: str,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        frame = extract_embeddings_dataframe(
            checkpoint=checkpoint,
            data_dir=tokenized_dir,
            split=split_name,
            split_dir=sampled_split_dir,
            batch_size=batch_size,
            device=device,
            repr_type=repr_type,
        )
        frame.to_parquet(output_dir / f"pragma_lite_{repr_type}_{split_name}_embeddings.parquet", index=False)
        result[split_name] = frame
    return result


def run_transxion_binary_benchmark(
    checkpoint: Path,
    tokenized_dir: Path,
    processed_dir: Path,
    split_dir: Path,
    output_dir: Path,
    sample_size: int,
    batch_size: int,
    device: str,
    seed: int,
    repr_type: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_ids = sample_split_ids(
        labels_path=processed_dir / "labels.parquet",
        split_dir=split_dir,
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
    )
    sampled_split_dir = output_dir / "sampled_splits"
    sampled_entity_ids = {entity_id for ids in sampled_ids.values() for entity_id in ids}

    raw_features = build_raw_feature_table(processed_dir, entity_ids=sampled_entity_ids)
    raw_splits = {
        split_name: raw_features[raw_features["entity_id"].isin(ids)].sort_values("entity_id").reset_index(drop=True)
        for split_name, ids in sampled_ids.items()
    }

    pragma_splits = _extract_split_embeddings(
        checkpoint=checkpoint,
        tokenized_dir=tokenized_dir,
        sampled_split_dir=sampled_split_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        repr_type=repr_type,
    )

    results = {
        "pragma_lite_logreg": run_pragma_lite_logreg(
            pragma_splits["train"],
            pragma_splits["valid"],
            pragma_splits["test"],
            seed=seed,
            output_dir=output_dir,
        ),
        "xgboost": run_xgboost_grid_search(
            raw_splits["train"],
            raw_splits["valid"],
            raw_splits["test"],
            seed=seed,
            output_dir=output_dir,
        ),
        "catboost": run_catboost_grid_search(
            raw_splits["train"],
            raw_splits["valid"],
            raw_splits["test"],
            seed=seed,
            output_dir=output_dir,
        ),
    }

    report = {
        "task": "transxion_binary_downstream_benchmark",
        "checkpoint": str(checkpoint),
        "tokenized_dir": str(tokenized_dir),
        "processed_dir": str(processed_dir),
        "split_dir": str(split_dir),
        "sampled_split_dir": str(sampled_split_dir),
        "sample_size": int(sum(len(ids) for ids in sampled_ids.values())),
        "requested_sample_size": int(sample_size),
        "repr_type": repr_type,
        "seed": seed,
        "results": results,
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(_safe_json(report), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenized_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repr_type", default="concat", choices=REPRESENTATION_TYPES)
    args = parser.parse_args()

    run_transxion_binary_benchmark(
        checkpoint=Path(args.checkpoint),
        tokenized_dir=Path(args.tokenized_dir),
        processed_dir=Path(args.processed_dir),
        split_dir=Path(args.split_dir),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        repr_type=args.repr_type,
    )


if __name__ == "__main__":
    main()
