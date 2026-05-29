from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.fs import ensure_dir, write_json
from src.tokenizer.vocab import TokenizerVocab


def _time_delta_bucket(delta_minutes: float | None) -> int:
    if delta_minutes is None or not np.isfinite(delta_minutes) or delta_minutes < 0:
        return 0
    x = float(delta_minutes)
    b = int(np.floor(np.log1p(x)))
    return int(np.clip(b, 0, 31))


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

    for entity_id, ev_df in tqdm(grouped, desc="encode", total=len(grouped)):
        if entity_id not in profile_map.index or entity_id not in label_map.index:
            continue

        tokens: list[int] = [vocab.usr_id]

        prof = profile_map.loc[entity_id]
        prof_tokens: list[int] = []
        for col in vocab.profile_cols:
            key = f"KP:{col}"
            prof_tokens.append(vocab.encode_token(key))
            val = prof[col] if col in prof.index else None
            col_key = f"P:{col}"
            if col_key in vocab.numeric_binners:
                b = vocab.numeric_binners[col_key].bucket(val)
                b = max(b, 0)
                prof_tokens.append(vocab.encode_token(f"VP:{col}#B{b}"))
            else:
                v = "[NA]" if pd.isna(val) else str(val)
                prof_tokens.append(vocab.encode_token(f"VP:{col}={v}"))

        prof_tokens = prof_tokens[: args.max_profile_tokens]
        tokens.extend(prof_tokens)

        ev_df = ev_df.sort_values("timestamp", ascending=True).tail(args.max_events)
        last_ts = None
        for _, row in ev_df.iterrows():
            event_tokens: list[int] = [vocab.evt_id]
            ts = row["timestamp"]
            if last_ts is None:
                delta_min = None
            else:
                delta_min = (ts - last_ts).total_seconds() / 60.0
            last_ts = ts

            dt_bucket = _time_delta_bucket(delta_min)
            event_tokens.append(vocab.encode_token("KE:time_delta"))
            event_tokens.append(vocab.encode_token(f"VE:time_delta#B{dt_bucket}"))

            for col in vocab.event_cols:
                key = f"KE:{col}"
                event_tokens.append(vocab.encode_token(key))
                val = row[col] if col in row.index else None
                col_key = f"E:{col}"
                if col_key in vocab.numeric_binners:
                    b = vocab.numeric_binners[col_key].bucket(val)
                    b = max(b, 0)
                    event_tokens.append(vocab.encode_token(f"VE:{col}#B{b}"))
                else:
                    v = "[NA]" if pd.isna(val) else str(val)
                    event_tokens.append(vocab.encode_token(f"VE:{col}={v}"))

                if len(event_tokens) >= args.max_event_tokens:
                    break

            tokens.extend(event_tokens[: args.max_event_tokens])

        attention_mask = [1] * len(tokens)
        rows.append(
            {
                "entity_id": int(entity_id),
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "label": int(label_map.loc[entity_id]["label"]),
                "evaluation_time": str(label_map.loc[entity_id]["evaluation_time"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "dataset.parquet", index=False)
    write_json(
        out_dir / "tokenized_summary.json",
        {"num_records": int(len(df)), "vocab_size": int(len(vocab.token_to_id)), "max_len": int(df["input_ids"].map(len).max()) if len(df) else 0},
    )


if __name__ == "__main__":
    main()
