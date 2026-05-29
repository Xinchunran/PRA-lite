from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.tokenizer.vocab import NumericBinner, SPECIAL_TOKENS, TokenizerVocab


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_buckets", type=int, default=100)
    parser.add_argument("--min_freq", type=int, default=5)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    events = pd.read_parquet(processed_dir / "events.parquet")

    profile_cols = [c for c in profiles.columns if c != "entity_id"]
    event_cols = [c for c in events.columns if c not in {"entity_id", "event_id", "timestamp"}]

    token_to_id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    next_id = len(token_to_id)

    numeric_binners: dict[str, NumericBinner] = {}

    def add_token(token: str) -> None:
        nonlocal next_id
        if token not in token_to_id:
            token_to_id[token] = next_id
            next_id += 1

    for col in profile_cols:
        add_token(f"KP:{col}")
        if _is_numeric(profiles[col]):
            values = profiles[col].dropna().astype("float64")
            if len(values) > 0:
                edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, args.num_buckets + 1)[1:-1]).tolist()
            else:
                edges = []
            numeric_binners[f"P:{col}"] = NumericBinner(edges=edges)
            for b in range(args.num_buckets + 1):
                add_token(f"VP:{col}#B{b}")
        else:
            counts = Counter(profiles[col].astype("string").fillna("[NA]").tolist())
            for v, c in counts.items():
                if c >= args.min_freq:
                    add_token(f"VP:{col}={v}")

    for col in event_cols:
        add_token(f"KE:{col}")
        if _is_numeric(events[col]):
            values = events[col].dropna().astype("float64")
            if len(values) > 0:
                edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, args.num_buckets + 1)[1:-1]).tolist()
            else:
                edges = []
            numeric_binners[f"E:{col}"] = NumericBinner(edges=edges)
            for b in range(args.num_buckets + 1):
                add_token(f"VE:{col}#B{b}")
        else:
            counts = Counter(events[col].astype("string").fillna("[NA]").tolist())
            for v, c in counts.items():
                if c >= args.min_freq:
                    add_token(f"VE:{col}={v}")

    add_token("KE:time_delta")
    for b in range(32):
        add_token(f"VE:time_delta#B{b}")

    vocab = TokenizerVocab(
        token_to_id=token_to_id,
        profile_cols=profile_cols,
        event_cols=event_cols,
        numeric_binners=numeric_binners,
    )
    vocab.save(out_dir)

    summary = {
        "vocab_size": len(token_to_id),
        "profile_cols": profile_cols,
        "event_cols": event_cols,
        "num_buckets": args.num_buckets,
        "min_freq": args.min_freq,
    }
    write_json(out_dir / "vocab_summary.json", summary)


if __name__ == "__main__":
    main()
