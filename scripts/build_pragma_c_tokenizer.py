#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.pragma_c.common import EVENT_COLS, PROFILE_COLS, build_account_event_view, compute_profile_state, history_before
from src.tokenizer.schema import FieldSchema, VocabBuildConfig
from src.tokenizer.text_bpe import train_text_tokenizer
from src.tokenizer.vocab import NumericBinner, SPECIAL_TOKENS, TokenizerVocab


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _normalize_categorical_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "[NA]"
    return str(value).strip().lower() or "[NA]"


def _infer_field_schema(
    namespace: str,
    col: str,
    series: pd.Series,
    cfg: VocabBuildConfig,
) -> FieldSchema:
    normalized_name = col.strip().lower()
    if _is_numeric(series):
        return FieldSchema(namespace=namespace, name=col, value_type="numeric")
    if any(token in normalized_name for token in cfg.force_textual_cols):
        cardinality = int(series.astype("string").dropna().nunique())
        return FieldSchema(namespace=namespace, name=col, value_type="textual", cardinality=cardinality)
    if any(token in normalized_name for token in cfg.force_categorical_cols):
        cardinality = int(series.astype("string").dropna().nunique())
        return FieldSchema(namespace=namespace, name=col, value_type="categorical", cardinality=cardinality)
    cardinality = int(series.astype("string").dropna().nunique())
    value_type = "categorical" if cardinality <= cfg.categorical_threshold else "textual"
    return FieldSchema(namespace=namespace, name=col, value_type=value_type, cardinality=cardinality)


def build_tokenizer(
    output_root: str | Path,
    *,
    num_buckets: int = 100,
    min_freq: int = 5,
    profile_sample_limit: int = 200000,
    max_history_events: int = 6500,
    seed: int = 42,
    categorical_threshold: int = 2048,
    max_text_vocab_size: int = 28000,
    max_value_tokens_per_field: int = 4,
    numeric_zero_bucket: bool = True,
) -> Path:
    output_root = Path(output_root)
    tokenizer_dir = ensure_dir(output_root / "tokenizer")
    build_cfg = VocabBuildConfig(
        num_numeric_bins=num_buckets,
        categorical_threshold=categorical_threshold,
        max_text_vocab_size=max_text_vocab_size,
        max_value_tokens_per_field=max_value_tokens_per_field,
        numeric_zero_bucket=numeric_zero_bucket,
    )
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
    field_value_types: dict[str, str] = {}
    categorical_values: dict[str, list[str]] = {}
    text_samples: list[str] = []

    def add_token(token: str) -> None:
        nonlocal next_id
        if token not in token_to_id:
            token_to_id[token] = next_id
            next_id += 1

    field_schemas: list[FieldSchema] = []
    for col in PROFILE_COLS:
        field_schemas.append(_infer_field_schema("P", col, profile_fit_df[col], build_cfg))
    for col in EVENT_COLS:
        field_schemas.append(_infer_field_schema("E", col, event_fit_df[col], build_cfg))

    schema_map = {(schema.namespace, schema.name): schema for schema in field_schemas}

    for col in PROFILE_COLS:
        add_token(f"K:P:{col}")
        schema = schema_map[("P", col)]
        field_key = f"P:{col}"
        field_value_types[field_key] = schema.value_type
        if schema.value_type == "numeric":
            values = pd.to_numeric(profile_fit_df[col], errors="coerce").dropna().astype("float64")
            if numeric_zero_bucket:
                values = values[values != 0.0]
            edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]).tolist() if len(values) > 0 else []
            numeric_binners[field_key] = NumericBinner(edges=edges)
            add_token(f"V:{field_key}#[NA]")
            if numeric_zero_bucket:
                add_token(f"V:{field_key}#ZERO")
            for bucket_idx in range(num_buckets + 1):
                add_token(f"V:{field_key}#B{bucket_idx}")
        elif schema.value_type == "categorical":
            counts = Counter(_normalize_categorical_value(value) for value in profile_fit_df[col].tolist())
            values = sorted(value for value, count in counts.items() if count >= min_freq and value != "[NA]")
            categorical_values[field_key] = values
            add_token(f"V:{field_key}=[UNK]")
            add_token(f"V:{field_key}=[NA]")
            for value in values:
                add_token(f"V:{field_key}={value}")
        else:
            text_samples.extend(str(value) for value in profile_fit_df[col].dropna().tolist() if str(value).strip())

    for col in EVENT_COLS:
        add_token(f"K:E:{col}")
        schema = schema_map[("E", col)]
        field_key = f"E:{col}"
        field_value_types[field_key] = schema.value_type
        if schema.value_type == "numeric":
            values = pd.to_numeric(event_fit_df[col], errors="coerce").dropna().astype("float64")
            if numeric_zero_bucket:
                values = values[values != 0.0]
            edges = np.quantile(values.to_numpy(), np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]).tolist() if len(values) > 0 else []
            numeric_binners[field_key] = NumericBinner(edges=edges)
            add_token(f"V:{field_key}#[NA]")
            if numeric_zero_bucket:
                add_token(f"V:{field_key}#ZERO")
            for bucket_idx in range(num_buckets + 1):
                add_token(f"V:{field_key}#B{bucket_idx}")
        elif schema.value_type == "categorical":
            counts = Counter(_normalize_categorical_value(value) for value in event_fit_df[col].tolist())
            values = sorted(value for value, count in counts.items() if count >= min_freq and value != "[NA]")
            categorical_values[field_key] = values
            add_token(f"V:{field_key}=[UNK]")
            add_token(f"V:{field_key}=[NA]")
            for value in values:
                add_token(f"V:{field_key}={value}")
        else:
            text_samples.extend(str(value) for value in event_fit_df[col].dropna().tolist() if str(value).strip())

    text_tokenizer_path: str | None = None
    text_tokenizer = None
    if any(value_type == "textual" for value_type in field_value_types.values()):
        add_token("T:[UNK]")
        add_token("T:[NA]")
        text_tokenizer_path = "text_bpe.json"
        text_tokenizer, _ = train_text_tokenizer(
            text_samples,
            tokenizer_dir / text_tokenizer_path,
            vocab_size=max_text_vocab_size,
        )
        pieces = []
        try:
            pieces = list(getattr(text_tokenizer, "get_vocab", lambda: {})().keys())
        except Exception:
            pieces = []
        if not pieces and hasattr(text_tokenizer, "vocab"):
            pieces = list(getattr(text_tokenizer, "vocab"))
        for piece in pieces:
            if piece != "[UNK]":
                add_token(f"T:{piece}")

    vocab = TokenizerVocab(
        token_to_id=token_to_id,
        profile_cols=list(PROFILE_COLS),
        event_cols=list(EVENT_COLS),
        numeric_binners=numeric_binners,
        field_value_types=field_value_types,
        categorical_values=categorical_values,
        text_tokenizer_path=text_tokenizer_path,
        max_value_tokens_per_field=max_value_tokens_per_field,
        tokenizer_version=2,
        numeric_zero_bucket=numeric_zero_bucket,
        text_tokenizer=text_tokenizer,
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
            "categorical_threshold": int(categorical_threshold),
            "max_text_vocab_size": int(max_text_vocab_size),
            "max_value_tokens_per_field": int(max_value_tokens_per_field),
            "numeric_zero_bucket": bool(numeric_zero_bucket),
        },
    )
    write_json(
        tokenizer_dir / "vocab_summary.json",
        {
            "vocab_size": len(token_to_id),
            "profile_cols": list(PROFILE_COLS),
            "event_cols": list(EVENT_COLS),
            "tokenizer_version": 2,
            "field_value_type_counts": dict(Counter(field_value_types.values())),
            "max_value_tokens_per_field": int(max_value_tokens_per_field),
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
    parser.add_argument("--categorical_threshold", type=int, default=2048)
    parser.add_argument("--max_text_vocab_size", type=int, default=28000)
    parser.add_argument("--max_value_tokens_per_field", type=int, default=4)
    parser.add_argument("--numeric_zero_bucket", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    build_tokenizer(
        args.output_root,
        num_buckets=args.num_buckets,
        min_freq=args.min_freq,
        profile_sample_limit=args.profile_sample_limit,
        max_history_events=args.max_history_events,
        seed=args.seed,
        categorical_threshold=args.categorical_threshold,
        max_text_vocab_size=args.max_text_vocab_size,
        max_value_tokens_per_field=args.max_value_tokens_per_field,
        numeric_zero_bucket=bool(args.numeric_zero_bucket),
    )


if __name__ == "__main__":
    main()
