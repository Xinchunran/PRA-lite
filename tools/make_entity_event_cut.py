from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir, write_json


def make_entity_event_cut(processed_dir: str | Path, output_dir: str | Path, target_events: int, seed: int = 26) -> Path:
    src = Path(processed_dir)
    out = ensure_dir(output_dir)

    profiles = pd.read_parquet(src / "profiles.parquet")
    events = pd.read_parquet(src / "events.parquet")
    labels = pd.read_parquet(src / "labels.parquet")

    counts = events.groupby("entity_id").size().reset_index(name="n_events")
    counts = counts.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    counts["cum_events"] = counts["n_events"].cumsum()
    selected = counts.loc[counts["cum_events"] <= target_events, "entity_id"]
    if selected.empty:
        selected = counts.head(1)["entity_id"]

    selected_ids = set(selected.tolist())
    profiles_cut = profiles[profiles["entity_id"].isin(selected_ids)].copy()
    events_cut = events[events["entity_id"].isin(selected_ids)].copy()
    labels_cut = labels[labels["entity_id"].isin(selected_ids)].copy()

    profiles_cut.to_parquet(out / "profiles.parquet", index=False)
    events_cut.to_parquet(out / "events.parquet", index=False)
    labels_cut.to_parquet(out / "labels.parquet", index=False)

    for name in ["schema.json", "vocab_stats.json"]:
        src_path = src / name
        if src_path.exists():
            (out / name).write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(
        out / "cut_summary.json",
        {
            "source_events": int(len(events)),
            "target_events": int(target_events),
            "selected_entities": int(len(selected_ids)),
            "cut_events": int(len(events_cut)),
            "cut_profiles": int(len(profiles_cut)),
            "cut_labels": int(len(labels_cut)),
            "seed": int(seed),
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()
    make_entity_event_cut(args.processed_dir, args.output_dir, args.target_events, seed=args.seed)


if __name__ == "__main__":
    main()
