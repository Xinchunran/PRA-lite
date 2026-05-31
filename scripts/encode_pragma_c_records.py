#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

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


def _load_shard_eval(
    output_root: Path,
    *,
    shard_index: int,
    num_shards: int,
    max_eval_points_per_account_train: int,
    max_eval_points_per_account_valid: int,
    max_eval_points_per_account_calibration: int,
) -> pd.DataFrame:
    shard_eval_path = output_root / "eval_points" / "encode_shards" / f"shard_{shard_index:05d}.parquet"
    if shard_eval_path.exists():
        shard_eval = pd.read_parquet(shard_eval_path)
        shard_eval["evaluation_time"] = pd.to_datetime(shard_eval["evaluation_time"], utc=True, errors="coerce")
        print(
            f"[pragma_c_encode] shard={shard_index:05d} loaded pre-indexed eval points rows={len(shard_eval)} "
            f"entities={shard_eval['entity_id'].nunique() if not shard_eval.empty else 0}",
            flush=True,
        )
        return shard_eval

    print(
        f"[pragma_c_encode] shard={shard_index:05d} no pre-indexed eval file found, falling back to full eval scan",
        flush=True,
    )
    eval_points = pd.read_parquet(output_root / "eval_points" / "eval_points.parquet")
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
    eval_points["encode_shard_index"] = eval_points["entity_id"].map(lambda value: stable_hash_bucket(value, num_shards))
    return eval_points[eval_points["encode_shard_index"] == int(shard_index)].copy()


def _load_transactions_for_entities(output_root: Path, entity_ids: list[int]) -> pd.DataFrame:
    tx_path = output_root / "canonical" / "transactions.parquet"
    if not entity_ids:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "transaction_time",
                "sender_entity_id",
                "receiver_entity_id",
                "from_bank",
                "to_bank",
                "amount_paid",
                "amount_received",
                "payment_currency",
                "receiving_currency",
                "payment_format",
                "is_laundering",
            ]
        )

    tx_columns = [
        "transaction_id",
        "transaction_time",
        "sender_entity_id",
        "receiver_entity_id",
        "from_bank",
        "to_bank",
        "amount_paid",
        "amount_received",
        "payment_currency",
        "receiving_currency",
        "payment_format",
        "is_laundering",
    ]
    entity_values = sorted({int(entity_id) for entity_id in entity_ids})
    filter_expr = ds.field("sender_entity_id").isin(entity_values) | ds.field("receiver_entity_id").isin(entity_values)
    tx = ds.dataset(tx_path).to_table(columns=tx_columns, filter=filter_expr).to_pandas()
    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")
    return tx


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
    started_at = time.perf_counter()
    out_dir = ensure_dir(output_root / "tokenized_shards" / f"shard_{shard_index:05d}")
    vocab = TokenizerVocab.load(output_root / "tokenizer")
    print(f"[pragma_c_encode] shard={shard_index:05d} stage=load_eval", flush=True)
    shard_eval = _load_shard_eval(
        output_root,
        shard_index=shard_index,
        num_shards=num_shards,
        max_eval_points_per_account_train=max_eval_points_per_account_train,
        max_eval_points_per_account_valid=max_eval_points_per_account_valid,
        max_eval_points_per_account_calibration=max_eval_points_per_account_calibration,
    )
    entity_ids = sorted({int(entity_id) for entity_id in shard_eval["entity_id"].unique().tolist()}) if not shard_eval.empty else []
    print(
        f"[pragma_c_encode] shard={shard_index:05d} stage=load_tx entities={len(entity_ids)} eval_rows={len(shard_eval)}",
        flush=True,
    )
    tx = _load_transactions_for_entities(output_root, entity_ids)
    print(
        f"[pragma_c_encode] shard={shard_index:05d} stage=build_event_view tx_rows={len(tx)} elapsed_s={time.perf_counter() - started_at:.1f}",
        flush=True,
    )
    account_events = build_account_event_view(tx)
    if entity_ids:
        account_events = account_events[account_events["entity_id"].isin(entity_ids)].copy()
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
    progress_every = 10_000

    try:
        ordered_eval = shard_eval.sort_values(["evaluation_time", "entity_id", "anchor_transaction_id"], kind="stable")
        total_rows = len(ordered_eval)
        print(
            f"[pragma_c_encode] shard={shard_index:05d} stage=encode total_rows={total_rows} entities={len(event_groups)}",
            flush=True,
        )
        for row_idx, row in enumerate(ordered_eval.itertuples(index=False), start=1):
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
            if row_idx % progress_every == 0:
                print(
                    f"[pragma_c_encode] shard={shard_index:05d} progress={row_idx}/{total_rows} "
                    f"elapsed_s={time.perf_counter() - started_at:.1f}",
                    flush=True,
                )
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
    print(
        f"[pragma_c_encode] shard={shard_index:05d} stage=done records={summary['num_records']} "
        f"elapsed_s={time.perf_counter() - started_at:.1f}",
        flush=True,
    )
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
