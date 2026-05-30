from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.common.fs import ensure_dir, write_json
from src.tokenizer.structured import StructuredRecordConfig, encode_record
from src.tokenizer.vocab import TokenizerVocab


_WORKER_VOCAB: TokenizerVocab | None = None
_WORKER_CFG: StructuredRecordConfig | None = None


def _init_worker(vocab: TokenizerVocab, cfg: StructuredRecordConfig) -> None:
    global _WORKER_VOCAB, _WORKER_CFG
    _WORKER_VOCAB = vocab
    _WORKER_CFG = cfg


def _encode_entity_payload(payload: tuple[int, dict[str, Any], list[dict[str, Any]], pd.Timestamp, int]) -> dict[str, Any]:
    entity_id, profile, event_rows, evaluation_time, label = payload
    if _WORKER_VOCAB is None or _WORKER_CFG is None:
        raise RuntimeError("Tokenizer worker not initialized")
    encoded = encode_record(
        vocab=_WORKER_VOCAB,
        profile=profile,
        events=event_rows,
        evaluation_time=evaluation_time,
        cfg=_WORKER_CFG,
    )
    return {
        "entity_id": int(entity_id),
        **encoded,
        "label": int(label),
        "evaluation_time": str(evaluation_time),
    }


def _make_payloads(
    profiles: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    max_events: int,
) -> list[tuple[int, dict[str, Any], list[dict[str, Any]], pd.Timestamp, int]]:
    profile_map = profiles.set_index("entity_id", drop=False).to_dict(orient="index")
    label_map = labels.set_index("entity_id", drop=False).to_dict(orient="index")
    grouped = events.groupby("entity_id", sort=False)

    payloads: list[tuple[int, dict[str, Any], list[dict[str, Any]], pd.Timestamp, int]] = []
    for entity_id, ev_df in grouped:
        profile = profile_map.get(entity_id)
        label_row = label_map.get(entity_id)
        if profile is None or label_row is None:
            continue
        evaluation_time = label_row["evaluation_time"]
        if pd.isna(evaluation_time):
            continue
        event_rows = ev_df.tail(max_events).to_dict("records")
        payloads.append((int(entity_id), profile, event_rows, evaluation_time, int(label_row["label"])))
    return payloads


def _resolve_mp_context(num_workers: int) -> multiprocessing.context.BaseContext | None:
    if num_workers <= 1:
        return None
    available = multiprocessing.get_all_start_methods()
    if sys.platform.startswith("linux") and "fork" in available:
        return multiprocessing.get_context("fork")
    if "forkserver" in available:
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context(available[0])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_events", type=int, default=512)
    parser.add_argument("--max_event_tokens", type=int, default=24)
    parser.add_argument("--max_profile_tokens", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=1)
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
    events = events.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    labels["evaluation_time"] = pd.to_datetime(labels["evaluation_time"], utc=True, errors="coerce")
    labels = labels.dropna(subset=["evaluation_time"]).copy()

    cfg = StructuredRecordConfig(
        max_events=args.max_events,
        max_event_tokens=args.max_event_tokens,
        max_profile_tokens=args.max_profile_tokens,
    )
    payloads = _make_payloads(profiles=profiles, events=events, labels=labels, max_events=args.max_events)

    rows: list[dict[str, Any]] = []
    num_workers = max(int(args.num_workers), 1)
    if num_workers == 1 or len(payloads) <= 1:
        _init_worker(vocab, cfg)
        for payload in tqdm(payloads, desc="encode", total=len(payloads)):
            rows.append(_encode_entity_payload(payload))
    else:
        mp_context = _resolve_mp_context(num_workers)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(vocab, cfg),
        ) as executor:
            for row in tqdm(
                executor.map(_encode_entity_payload, payloads, chunksize=32),
                desc="encode",
                total=len(payloads),
            ):
                rows.append(row)

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
