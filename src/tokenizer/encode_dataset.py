from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.fs import ensure_dir, write_json
from src.tokenizer.time import periodic_encode, soft_log_seconds
from src.tokenizer.vocab import TokenizerVocab


def _calendar_features(ts: pd.Timestamp) -> list[float]:
    hour = periodic_encode(np.array([ts.hour], dtype=np.float64), period=24.0)[0]
    dow = periodic_encode(np.array([ts.dayofweek], dtype=np.float64), period=7.0)[0]
    dom = periodic_encode(np.array([ts.day], dtype=np.float64), period=31.0)[0]
    feats = np.concatenate([hour, dow, dom], axis=0)
    return feats.astype(np.float32).tolist()


def _event_time_feature(event_ts: pd.Timestamp, evaluation_ts: pd.Timestamp) -> float:
    delta_seconds = max(0.0, (evaluation_ts - event_ts).total_seconds())
    return float(soft_log_seconds(delta_seconds))


def _pad_1d(values: list[int], length: int, pad_value: int = 0) -> tuple[list[int], list[int]]:
    clipped = values[:length]
    mask = [1] * len(clipped) + [0] * max(length - len(clipped), 0)
    padded = clipped + [pad_value] * max(length - len(clipped), 0)
    return padded, mask


def _pad_2d(
    rows: list[list[int]],
    outer: int,
    inner: int,
    pad_value: int = 0,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    padded_rows: list[list[int]] = []
    masks: list[list[int]] = []
    row_mask: list[int] = []
    for row in rows[:outer]:
        padded_row, row_attn = _pad_1d(row, inner, pad_value=pad_value)
        padded_rows.append(padded_row)
        masks.append(row_attn)
        row_mask.append(1)
    while len(padded_rows) < outer:
        padded_rows.append([pad_value] * inner)
        masks.append([0] * inner)
        row_mask.append(0)
    return padded_rows, masks, row_mask


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

        prof = profile_map.loc[entity_id]
        prof_tokens: list[int] = []
        for col in vocab.profile_cols:
            prof_tokens.append(vocab.encode_token(f"KP:{col}"))
            val = prof[col] if col in prof.index else None
            col_key = f"P:{col}"
            if col_key in vocab.numeric_binners:
                b = vocab.numeric_binners[col_key].bucket(val)
                b = max(b, 0)
                prof_tokens.append(vocab.encode_token(f"VP:{col}#B{b}"))
            else:
                v = "[NA]" if pd.isna(val) else str(val)
                prof_tokens.append(vocab.encode_token(f"VP:{col}={v}"))

        prof_input_ids, prof_mask = _pad_1d(prof_tokens, args.max_profile_tokens, pad_value=vocab.pad_id)

        flat_tokens: list[int] = [vocab.usr_id]
        flat_tokens.extend(prof_tokens[: args.max_profile_tokens])

        evaluation_time = pd.to_datetime(label_map.loc[entity_id]["evaluation_time"], utc=True, errors="coerce")
        if pd.isna(evaluation_time):
            continue
        ev_df = ev_df.sort_values("timestamp", ascending=True).tail(args.max_events)
        event_rows: list[list[int]] = []
        event_times: list[float] = []
        calendar_rows: list[list[float]] = []
        for _, row in ev_df.iterrows():
            event_tokens: list[int] = [vocab.evt_id]
            ts = row["timestamp"]
            for col in vocab.event_cols:
                event_tokens.append(vocab.encode_token(f"KE:{col}"))
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
            event_tokens = event_tokens[: args.max_event_tokens]
            event_rows.append(event_tokens)
            event_times.append(_event_time_feature(ts, evaluation_time))
            calendar_rows.append(_calendar_features(ts))
            flat_tokens.extend(event_tokens)

        event_input_ids, event_attention_mask, event_mask = _pad_2d(
            event_rows,
            outer=args.max_events,
            inner=args.max_event_tokens,
            pad_value=vocab.pad_id,
        )
        event_times = event_times[: args.max_events] + [0.0] * max(args.max_events - len(event_times), 0)
        calendar_rows = calendar_rows[: args.max_events] + [[0.0] * 6] * max(args.max_events - len(calendar_rows), 0)
        attention_mask = [1] * len(flat_tokens)
        rows.append(
            {
                "entity_id": int(entity_id),
                "input_ids": flat_tokens,
                "attention_mask": attention_mask,
                "profile_input_ids": prof_input_ids,
                "profile_attention_mask": prof_mask,
                "event_input_ids": event_input_ids,
                "event_attention_mask": event_attention_mask,
                "event_times": event_times,
                "calendar_features": calendar_rows,
                "event_mask": event_mask,
                "label": int(label_map.loc[entity_id]["label"]),
                "evaluation_time": str(evaluation_time),
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
