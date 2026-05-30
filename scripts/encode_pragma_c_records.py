#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir, write_json
from src.common.tokenized_lmdb import TokenizedLmdbWriter
from src.pragma_c.common import (
    PRETRAIN_EVAL_SOURCES,
    STAGE_C_SPLITS,
    apply_split_caps,
    build_account_event_view,
    compute_profile_state,
    history_before,
    stable_hash_bucket,
)
from src.tokenizer.structured import StructuredRecordConfig, encode_record
from src.tokenizer.vocab import TokenizerVocab


def _build_writers(out_dir: Path, map_size_gb: int, commit_interval: int) -> tuple[TokenizedLmdbWriter, dict[str, TokenizedLmdbWriter]]:
    dataset_writer = TokenizedLmdbWriter(out_dir / "dataset.lmdb", map_size_gb=map_size_gb, commit_interval=commit_interval)
    split_writers = {
        split_name: TokenizedLmdbWriter(
            out_dir / f"{split_name}.lmdb",
            map_size_gb=map_size_gb,
            commit_interval=commit_interval,
        )
        for split_name in STAGE_C_SPLITS
    }
    return dataset_writer, split_writers


def encode_shard(
    output_root: str | Path,
    *,
    shard_index: int,
    num_shards: int,
    max_events: int,
    max_event_tokens: int,
    max_profile_tokens: int,
    max_history_events: int,
    max_eval_points_per_account_train: int,
    max_eval_points_per_account_valid: int,
    max_eval_points_per_account_calibration: int,
    lmdb_map_size_gb: int,
    lmdb_commit_interval: int,
) -> Path:
    output_root = Path(output_root)
    out_dir = ensure_dir(output_root / "tokenized_shards" / f"shard_{shard_index:05d}")
    vocab = TokenizerVocab.load(output_root / "tokenizer")
    tx = pd.read_parquet(output_root / "canonical" / "transactions.parquet")
    eval_points = pd.read_parquet(output_root / "eval_points" / "eval_points.parquet")

    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")
    eval_points["evaluation_time"] = pd.to_datetime(eval_points["evaluation_time"], utc=True, errors="coerce")
    eval_points = eval_points[
        (eval_points["task"] == "pretrain") & (eval_points["eval_source"].isin(PRETRAIN_EVAL_SOURCES))
    ].copy()

    split_caps = {
        "train": max_eval_points_per_account_train,
        "valid": max_eval_points_per_account_valid,
        "calibration": max_eval_points_per_account_calibration,
        "test": 0,
        "embargo": 0,
    }
    eval_points = apply_split_caps(eval_points, split_caps)
    eval_points["encode_shard_index"] = eval_points["eval_id"].map(lambda value: stable_hash_bucket(value, num_shards))
    shard_eval = eval_points[eval_points["encode_shard_index"] == int(shard_index)].copy()

    account_events = build_account_event_view(tx)
    event_groups = {int(entity_id): df.reset_index(drop=True) for entity_id, df in account_events.groupby("entity_id", sort=False)}
    cfg = StructuredRecordConfig(
        max_events=max_events,
        max_event_tokens=max_event_tokens,
        max_profile_tokens=max_profile_tokens,
    )

    dataset_writer, split_writers = _build_writers(out_dir, lmdb_map_size_gb, lmdb_commit_interval)
    counts = {split_name: 0 for split_name in STAGE_C_SPLITS}
    history_lengths: list[int] = []
    token_lengths: list[int] = []
    empty_histories = 0

    try:
        for row in shard_eval.sort_values(["evaluation_time", "entity_id", "anchor_transaction_id"], kind="stable").itertuples(index=False):
            entity_events = event_groups.get(int(row.entity_id))
            if entity_events is None:
                history = pd.DataFrame(columns=account_events.columns)
            else:
                history = history_before(entity_events, row.evaluation_time, max_history_events=max_history_events)
            profile = compute_profile_state(history, row.evaluation_time)
            encoded = encode_record(
                vocab=vocab,
                profile=profile,
                events=history,
                evaluation_time=row.evaluation_time,
                cfg=cfg,
            )
            history_count = int(len(history))
            encoded.update(
                {
                    "entity_id": int(row.entity_id),
                    "label": int(row.label),
                    "evaluation_time": row.evaluation_time.isoformat(),
                    "anchor_transaction_id": int(row.anchor_transaction_id),
                    "eval_source": str(row.eval_source),
                    "split": str(row.split),
                    "history_event_count": history_count,
                }
            )
            dataset_writer.write(encoded)
            split_writers[str(row.split)].write(encoded)
            counts[str(row.split)] += 1
            history_lengths.append(history_count)
            token_lengths.append(int(sum(sum(mask_row) for mask_row in encoded["event_token_mask"])))
            if history_count == 0:
                empty_histories += 1
    finally:
        dataset_writer.close()
        for writer in split_writers.values():
            writer.close()

    summary = {
        "stage": "C",
        "shard_id": f"shard_{shard_index:05d}",
        "split_mode": "pragma_c",
        "num_shards": int(num_shards),
        "counts": counts,
        "num_records": int(sum(counts.values())),
        "max_events": int(max_events),
        "max_event_tokens": int(max_event_tokens),
        "max_profile_tokens": int(max_profile_tokens),
        "vocab_size": len(vocab.token_to_id),
        "history_length_mean": float(sum(history_lengths) / len(history_lengths)) if history_lengths else 0.0,
        "history_length_max": int(max(history_lengths)) if history_lengths else 0,
        "token_length_mean": float(sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
        "empty_history_records": int(empty_histories),
    }
    write_json(out_dir / "shard_summary.json", summary)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=128)
    parser.add_argument("--max_events", type=int, default=256)
    parser.add_argument("--max_event_tokens", type=int, default=24)
    parser.add_argument("--max_profile_tokens", type=int, default=200)
    parser.add_argument("--max_history_events", type=int, default=6500)
    parser.add_argument("--max_eval_points_per_account_train", type=int, default=64)
    parser.add_argument("--max_eval_points_per_account_valid", type=int, default=32)
    parser.add_argument("--max_eval_points_per_account_calibration", type=int, default=32)
    parser.add_argument("--lmdb_map_size_gb", type=int, default=64)
    parser.add_argument("--lmdb_commit_interval", type=int, default=1024)
    args = parser.parse_args()
    encode_shard(
        args.output_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_events=args.max_events,
        max_event_tokens=args.max_event_tokens,
        max_profile_tokens=args.max_profile_tokens,
        max_history_events=args.max_history_events,
        max_eval_points_per_account_train=args.max_eval_points_per_account_train,
        max_eval_points_per_account_valid=args.max_eval_points_per_account_valid,
        max_eval_points_per_account_calibration=args.max_eval_points_per_account_calibration,
        lmdb_map_size_gb=args.lmdb_map_size_gb,
        lmdb_commit_interval=args.lmdb_commit_interval,
    )


if __name__ == "__main__":
    main()
