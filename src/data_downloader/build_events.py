from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.common.yaml_utils import load_yaml


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _compute_vocab_stats(profiles: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"profile": {}, "event": {}}

    for col in profiles.columns:
        if col in {"entity_id"}:
            continue
        if pd.api.types.is_numeric_dtype(profiles[col]):
            stats["profile"][col] = {"type": "numeric", "count": int(profiles[col].notna().sum())}
        else:
            stats["profile"][col] = {
                "type": "categorical",
                "unique": int(profiles[col].astype("string").fillna("[NA]").nunique()),
            }

    for col in events.columns:
        if col in {"entity_id", "event_id", "timestamp"}:
            continue
        if pd.api.types.is_numeric_dtype(events[col]):
            stats["event"][col] = {"type": "numeric", "count": int(events[col].notna().sum())}
        else:
            stats["event"][col] = {
                "type": "categorical",
                "unique": int(events[col].astype("string").fillna("[NA]").nunique()),
            }

    return stats


def build_transxion_events(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    ensure_dir(processed_dir)

    accounts = _safe_read_csv(raw_dir / "accounts.csv")
    persons = _safe_read_csv(raw_dir / "persons.csv")
    merchants = _safe_read_csv(raw_dir / "merchants.csv")
    transactions = _safe_read_csv(raw_dir / cfg["transaction_file"])

    accounts["account_id"] = accounts["account_id"].astype("int64")
    persons["person_id"] = persons["person_id"].astype("int64")

    profiles = accounts.merge(persons, on="person_id", how="left")
    profiles = profiles.rename(columns={"account_id": "entity_id"})
    profiles["entity_id"] = profiles["entity_id"].astype("int64")

    profile_cols = ["entity_id"]
    for c in ["entity_type", "age_bucket", "region", "account_region", "created_at"]:
        if c in profiles.columns:
            profile_cols.append(c)
    profiles = profiles[profile_cols]

    transactions["sender_id"] = transactions["sender_id"].astype("int64")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="coerce")
    transactions = transactions.dropna(subset=["timestamp"])

    events = transactions.copy()
    events = events.rename(columns={"sender_id": "entity_id"})
    events["event_id"] = events["transaction_id"].astype("int64")
    events = events.drop(columns=["transaction_id"])

    if "receiver_id" in events.columns:
        events["receiver_type"] = (events["receiver_id"].astype("int64") >= 10_000_000).map(
            {True: "merchant", False: "account"}
        )
    else:
        events["receiver_type"] = "unknown"

    events["event_type"] = "transaction"
    event_cols = ["entity_id", "event_id", "timestamp", "event_type"]
    for col in cfg.get("transaction_columns", []):
        if col in {"timestamp", "sender_id"}:
            continue
        if col in events.columns and col not in event_cols:
            event_cols.append(col)
    if "receiver_type" not in event_cols:
        event_cols.append("receiver_type")
    events = events[event_cols].sort_values(["entity_id", "timestamp", "event_id"]).reset_index(drop=True)

    labels = events.groupby("entity_id", as_index=False)[cfg["label_col"]].max()
    labels = labels.rename(columns={cfg["label_col"]: "label"})

    evaluation_time = events.groupby("entity_id", as_index=False)["timestamp"].max().rename(
        columns={"timestamp": "evaluation_time"}
    )
    labels = labels.merge(evaluation_time, on="entity_id", how="left")
    labels["evaluation_time"] = labels["evaluation_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    events_out = processed_dir / "events.parquet"
    profiles_out = processed_dir / "profiles.parquet"
    labels_out = processed_dir / "labels.parquet"

    events.to_parquet(events_out, index=False)
    profiles.to_parquet(profiles_out, index=False)
    labels.to_parquet(labels_out, index=False)

    schema = {
        "dataset": cfg["dataset"],
        "processed_dir": str(processed_dir),
        "profile_columns": [c for c in profiles.columns if c != "entity_id"],
        "event_columns": [c for c in events.columns if c not in {"entity_id", "event_id"}],
        "label_column": "label",
        "evaluation_time_column": "evaluation_time",
        "entity_id_dtype": "int64",
        "merchant_table_present": bool(len(merchants) > 0),
    }
    write_json(processed_dir / "schema.json", schema)
    write_json(processed_dir / "vocab_stats.json", _compute_vocab_stats(profiles, events))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    if cfg["dataset"] != "transxion":
        raise ValueError(f"Only transxion is supported in build_events right now, got: {cfg['dataset']}")
    build_transxion_events(config_path)


if __name__ == "__main__":
    main()
