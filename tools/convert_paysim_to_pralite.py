from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json


def convert_paysim_to_pralite(raw_csv: str | Path, processed_dir: str | Path, max_events: int = 0) -> Path:
    out_dir = ensure_dir(processed_dir)

    usecols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    df = pd.read_csv(raw_csv, usecols=usecols)
    df = df.sort_values(["step", "nameOrig", "nameDest"], kind="stable").reset_index(drop=True)
    if max_events > 0:
        df = df.head(max_events).copy()

    accounts = pd.Index(df["nameOrig"].astype("string").fillna("[NA]").unique())
    account_to_id = pd.Series(np.arange(len(accounts), dtype=np.int64), index=accounts)

    df["entity_id"] = df["nameOrig"].astype("string").fillna("[NA]").map(account_to_id).astype("int64")
    df["event_id"] = np.arange(len(df), dtype=np.int64)
    df["timestamp"] = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(df["step"].astype(int), unit="h")
    df["dest_type"] = df["nameDest"].astype(str).str[0].map({"C": "customer", "M": "merchant"}).fillna("unknown")

    grouped = df.groupby("entity_id", sort=False)
    profiles = pd.DataFrame(
        {
            "entity_id": grouped.size().index.astype("int64"),
            "account_role": "origin",
            "first_seen_step": grouped["step"].min().to_numpy(),
            "last_seen_step": grouped["step"].max().to_numpy(),
            "tx_count": grouped.size().to_numpy(),
            "mean_amount": grouped["amount"].mean().to_numpy(),
            "initial_oldbalance": grouped["oldbalanceOrg"].first().to_numpy(),
            "dominant_tx_type": grouped["type"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNK")
            .to_numpy(),
        }
    )

    events = df[
        [
            "entity_id",
            "event_id",
            "timestamp",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "dest_type",
        ]
    ].copy()
    events = events.rename(columns={"type": "event_type"})
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events = events.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)

    labels = grouped["isFraud"].max().reset_index().rename(columns={"isFraud": "label"})
    evaluation_time = events.groupby("entity_id", as_index=False)["timestamp"].max().rename(
        columns={"timestamp": "evaluation_time"}
    )
    labels = labels.merge(evaluation_time, on="entity_id", how="left")
    labels["evaluation_time"] = labels["evaluation_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    profiles.to_parquet(out_dir / "profiles.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    write_json(
        out_dir / "schema.json",
        {
            "dataset": "paysim",
            "processed_dir": str(out_dir),
            "profile_columns": [c for c in profiles.columns if c != "entity_id"],
            "event_columns": [c for c in events.columns if c not in {"entity_id", "event_id"}],
            "label_column": "label",
            "evaluation_time_column": "evaluation_time",
            "notes": "Derived profiles from origin account histories; isFraud and isFlaggedFraud excluded from events.",
        },
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="data/raw/paysim/transactions.csv")
    parser.add_argument("--processed_dir", default="data/processed/paysim_full")
    parser.add_argument("--max_events", type=int, default=0, help="Optional row cap before processing; 0 means full")
    args = parser.parse_args()
    convert_paysim_to_pralite(args.raw_csv, args.processed_dir, max_events=args.max_events)


if __name__ == "__main__":
    main()
