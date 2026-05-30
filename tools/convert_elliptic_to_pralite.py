from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json


def find_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def convert_elliptic_to_pralite(raw_dir: str | Path, processed_dir: str | Path) -> Path:
    raw_path = Path(raw_dir)
    out_dir = ensure_dir(processed_dir)

    features_path = find_file(raw_path, "elliptic_txs_features.csv")
    classes_path = find_file(raw_path, "elliptic_txs_classes.csv")
    edges_path = find_file(raw_path, "elliptic_txs_edgelist.csv")

    feature_cols = ["txId", "timestep"] + [f"f{i}" for i in range(165)]
    features = pd.read_csv(features_path, header=None, names=feature_cols).copy()
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    classes.columns = ["txId", "class"]
    edges.columns = ["txId1", "txId2"]

    features["txId"] = features["txId"].astype(str)
    classes["txId"] = classes["txId"].astype(str)
    edges["txId1"] = edges["txId1"].astype(str)
    edges["txId2"] = edges["txId2"].astype(str)

    id_map = pd.Series(np.arange(len(features), dtype=np.int64), index=features["txId"])
    features = features.assign(entity_id=features["txId"].map(id_map).astype("int64"))
    timestep_map = features.set_index("txId")["timestep"]

    profiles = features[["entity_id", "timestep"] + [f"f{i}" for i in range(165)]].copy()

    edges = edges[edges["txId1"].isin(id_map.index) & edges["txId2"].isin(id_map.index)].copy()
    src_events = pd.DataFrame(
        {
            "entity_id": edges["txId1"].map(id_map).astype("int64"),
            "counterparty_id": edges["txId2"].map(id_map).astype("int64"),
            "direction": "out",
            "neighbor_timestep": edges["txId2"].map(timestep_map).astype("int64"),
        }
    )
    dst_events = pd.DataFrame(
        {
            "entity_id": edges["txId2"].map(id_map).astype("int64"),
            "counterparty_id": edges["txId1"].map(id_map).astype("int64"),
            "direction": "in",
            "neighbor_timestep": edges["txId1"].map(timestep_map).astype("int64"),
        }
    )
    events = pd.concat([src_events, dst_events], ignore_index=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    events["event_type"] = "bitcoin_edge"
    events["amount"] = 1.0
    events["timestamp"] = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(
        events["neighbor_timestep"].astype(int), unit="D"
    )
    events = events[
        ["entity_id", "event_id", "timestamp", "event_type", "direction", "counterparty_id", "neighbor_timestep", "amount"]
    ].copy()
    events = events.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)

    labels = classes.merge(features[["txId", "entity_id", "timestep"]], on="txId", how="inner")
    labels = labels[labels["class"].astype(str).isin(["1", "2"])].copy()
    labels["label"] = (labels["class"].astype(str) == "1").astype(int)
    labels["evaluation_time"] = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(
        labels["timestep"].astype(int), unit="D"
    )
    labels["evaluation_time"] = labels["evaluation_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    labels = labels[["entity_id", "label", "evaluation_time"]]

    labeled_ids = set(labels["entity_id"].tolist())
    profiles = profiles[profiles["entity_id"].isin(labeled_ids)].copy()
    events = events[events["entity_id"].isin(labeled_ids)].copy()

    profiles.to_parquet(out_dir / "profiles.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    write_json(
        out_dir / "schema.json",
        {
            "dataset": "elliptic_200k",
            "processed_dir": str(out_dir),
            "profile_columns": [c for c in profiles.columns if c != "entity_id"],
            "event_columns": [c for c in events.columns if c not in {"entity_id", "event_id"}],
            "label_column": "label",
            "evaluation_time_column": "evaluation_time",
            "notes": "Graph-to-PRA-lite adaptation with unknown labels excluded from supervised benchmark.",
        },
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/elliptic")
    parser.add_argument("--processed_dir", default="data/processed/elliptic_200k")
    args = parser.parse_args()
    convert_elliptic_to_pralite(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
