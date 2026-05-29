from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.fs import ensure_dir, write_json
from src.tokenizer.structured import StructuredRecordConfig, encode_record
from src.tokenizer.vocab import TokenizerVocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_events", type=int, default=512)
    parser.add_argument("--max_event_tokens", type=int, default=24)
    parser.add_argument("--max_profile_tokens", type=int, default=200)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    vocab = TokenizerVocab.load(args.tokenizer_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    events = pd.read_parquet(processed_dir / "events.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")[["entity_id", "label", "evaluation_time"]]

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"])

    profile_map = profiles.set_index("entity_id", drop=False)
    label_map = labels.set_index("entity_id", drop=False)

    grouped = events.groupby("entity_id", sort=False)
    rows: list[dict] = []
    cfg = StructuredRecordConfig(
        max_events=args.max_events,
        max_event_tokens=args.max_event_tokens,
        max_profile_tokens=args.max_profile_tokens,
    )

    for entity_id, ev_df in tqdm(grouped, desc="encode", total=len(grouped)):
        if entity_id not in profile_map.index or entity_id not in label_map.index:
            continue

        prof = profile_map.loc[entity_id]
        evaluation_time = pd.to_datetime(label_map.loc[entity_id]["evaluation_time"], utc=True, errors="coerce")
        if pd.isna(evaluation_time):
            continue
        ev_df = ev_df.sort_values("timestamp", ascending=True).tail(args.max_events)
        encoded = encode_record(
            vocab=vocab,
            profile=prof,
            events=ev_df,
            evaluation_time=evaluation_time,
            cfg=cfg,
        )
        rows.append(
            {
                "entity_id": int(entity_id),
                **encoded,
                "label": int(label_map.loc[entity_id]["label"]),
                "evaluation_time": str(evaluation_time),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "dataset.parquet", index=False)
    write_json(
        out_dir / "tokenized_summary.json",
        {
            "num_records": int(len(df)),
            "vocab_size": int(len(vocab.token_to_id)),
            "max_profile_tokens": int(args.max_profile_tokens),
            "max_event_tokens": int(args.max_event_tokens),
            "max_events": int(args.max_events),
        },
    )


if __name__ == "__main__":
    main()
