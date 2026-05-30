#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.pragma_c.common import EVENT_COLS, PROFILE_COLS, build_account_event_view, compute_profile_state, history_before
from src.tokenizer.vocab import NumericBinner, SPECIAL_TOKENS, TokenizerVocab


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def build_tokenizer(
    output_root: str | Path,
    *,
    num_buckets: int = 100,
    min_freq: int = 5,
    profile_sample_limit: int = 200000,
    max_history_events: int = 6500,
    seed: int = 42,
) -> Path:
    output_root = Path(output_root)
    tokenizer_dir = ensure_dir(output_root / "tokenizer")
    tx = pd.read_parquet(output_root / "canonical" / "transactions.parquet")
    eval_points = pd.read_parquet(output_root / "eval_points" / "eval_points.parquet")

    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    train_eval = eval_points[(eval_points["task"] == "pretrain") & (eval_points["split"] == "train")].copy()
    if train_eval.empty:
        raise RuntimeError("No train pretrain eval points available for tokenizer fitting.")

    if len(train_eval) > profile_sample_limit:
        train_eval = train_eval.sample(n=profile_sample_limit, random_state=seed).sort_values(
            ["evaluation_time", "entity_id", "anchor_transaction_id"],
            kind="stable",
        )

    account_events = build_account_event_view(tx)
    train_end = train_eval["evaluation_time"].max()
    event_fit_df = account_events[account_events["timestamp"] < train_end][EVENT_COLS].copy()

    event_groups = {int(entity_id): df.reset_index(drop=True) for entity_id, df in account_events.groupby("entity_id", sort=False)}
    profile_rows: list[dict[str, object]] = []
    for row in train_eval.itertuples(index=False):
        entity_events = event_groups.get(int(row.entity_id))
        if entity_events is None:
            profile_rows.append(compute_profile_state(entity_events if entity_events is not None else pd.DataFrame(), row.evaluation_time))
            continue
        history = history_before(entity_events, row.evaluation_time, max_history_events=max_history_events)
        profile_rows.append(compute_profile_state(history, row.evaluation_time))
    profile_fit_df = pd.DataFrame(profile_rows, columns=PROFILE_COLS)

    token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    next_id = len(token_to_id)
    numeric_binners: dict[str, NumericBinner] = {}

    def add_token(token: str) -> None:
        nonlocal next_id
        if token not in token_to_id:
            token_to_id[token] = next_id
            next_id += 1

    for col in PROFILE_COLS:
        add_token(f"K:P:{col}")
        if _is_numeric(profile_fit_df[col]):
            values = pd.to_numeric(profile_fit_df[col], errors="coerce").dropna().astype("float64")
            edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]).tolist() if len(values) > 0 else []
            numeric_binners[f"P:{col}"] = NumericBinner(edges=edges)
            for bucket_idx in range(num_buckets + 1):
                add_token(f"V:P:{col}#B{bucket_idx}")
        else:
            counts = Counter(profile_fit_df[col].astype("string").fillna("UNK").tolist())
            for value, count in counts.items():
                if count >= min_freq:
                    add_token(f"V:P:{col}={value}")

    for col in EVENT_COLS:
        add_token(f"K:E:{col}")
        if _is_numeric(event_fit_df[col]):
            values = pd.to_numeric(event_fit_df[col], errors="coerce").dropna().astype("float64")
            edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]).tolist() if len(values) > 0 else []
            numeric_binners[f"E:{col}"] = NumericBinner(edges=edges)
            for bucket_idx in range(num_buckets + 1):
                add_token(f"V:E:{col}#B{bucket_idx}")
        else:
            counts = Counter(event_fit_df[col].astype("string").fillna("UNK").tolist())
            for value, count in counts.items():
                if count >= min_freq:
                    add_token(f"V:E:{col}={value}")

    vocab = TokenizerVocab(
        token_to_id=token_to_id,
        profile_cols=list(PROFILE_COLS),
        event_cols=list(EVENT_COLS),
        numeric_binners=numeric_binners,
    )
    vocab.save(tokenizer_dir)
    write_json(
        tokenizer_dir / "build_config.json",
        {
            "fit_split": "train",
            "profile_sample_limit": int(profile_sample_limit),
            "max_history_events": int(max_history_events),
            "num_buckets": int(num_buckets),
            "min_freq": int(min_freq),
            "seed": int(seed),
        },
    )
    write_json(
        tokenizer_dir / "vocab_summary.json",
        {
            "vocab_size": len(token_to_id),
            "profile_cols": list(PROFILE_COLS),
            "event_cols": list(EVENT_COLS),
        },
    )
    return tokenizer_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    parser.add_argument("--num_buckets", type=int, default=100)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--profile_sample_limit", type=int, default=200000)
    parser.add_argument("--max_history_events", type=int, default=6500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_tokenizer(
        args.output_root,
        num_buckets=args.num_buckets,
        min_freq=args.min_freq,
        profile_sample_limit=args.profile_sample_limit,
        max_history_events=args.max_history_events,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
