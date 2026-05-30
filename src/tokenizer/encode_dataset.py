from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.common.fs import ensure_dir, write_json
from src.common.tokenized_lmdb import TokenizedLmdbWriter
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


def _read_ids(path: Path) -> set[int]:
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _write_parquet(df: pd.DataFrame, path: Path, row_group_size: int) -> None:
    df.to_parquet(path, index=False, row_group_size=row_group_size)


def _write_split_parquets(df: pd.DataFrame, split_dir: Path, out_dir: Path, row_group_size: int) -> None:
    for split_name in ("train", "valid", "test"):
        ids_path = split_dir / f"{split_name}_ids.txt"
        if not ids_path.exists():
            continue
        split_ids = _read_ids(ids_path)
        split_df = df[df["entity_id"].isin(split_ids)].copy()
        _write_parquet(split_df, out_dir / f"{split_name}.parquet", row_group_size=row_group_size)


def _load_split_ids(split_dir: Path | None) -> dict[str, set[int]]:
    if split_dir is None:
        return {}
    split_ids: dict[str, set[int]] = {}
    for split_name in ("train", "valid", "test"):
        ids_path = split_dir / f"{split_name}_ids.txt"
        if ids_path.exists():
            split_ids[split_name] = _read_ids(ids_path)
    return split_ids


def _hash_split_name(entity_id: int, seed: int, train_frac: float, valid_frac: float) -> str:
    key = f"{seed}:{int(entity_id)}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    score = int.from_bytes(digest, byteorder="big", signed=False) / float(2**64)
    if score < train_frac:
        return "train"
    if score < train_frac + valid_frac:
        return "valid"
    return "test"


def _build_lmdb_writers(
    out_dir: Path,
    backend: str,
    split_ids: dict[str, set[int]],
    use_hash_split: bool,
    map_size_gb: int,
    commit_interval: int,
) -> tuple[TokenizedLmdbWriter | None, dict[str, TokenizedLmdbWriter]]:
    dataset_writer = None
    if backend in {"lmdb", "both"}:
        dataset_writer = TokenizedLmdbWriter(out_dir / "dataset.lmdb", map_size_gb=map_size_gb, commit_interval=commit_interval)
    split_writers: dict[str, TokenizedLmdbWriter] = {}
    if backend in {"lmdb", "both"} and (split_ids or use_hash_split):
        split_names = split_ids.keys() if split_ids else ("train", "valid", "test")
        for split_name in split_names:
            split_writers[split_name] = TokenizedLmdbWriter(
                out_dir / f"{split_name}.lmdb",
                map_size_gb=map_size_gb,
                commit_interval=commit_interval,
            )
    return dataset_writer, split_writers


def _route_row_to_lmdb(
    row: dict[str, Any],
    dataset_writer: TokenizedLmdbWriter | None,
    split_writers: dict[str, TokenizedLmdbWriter],
    split_ids: dict[str, set[int]],
    hash_split_seed: int | None,
    train_frac: float,
    valid_frac: float,
) -> None:
    if dataset_writer is not None:
        dataset_writer.write(row)
    entity_id = int(row["entity_id"])
    if hash_split_seed is not None:
        split_name = _hash_split_name(entity_id, hash_split_seed, train_frac, valid_frac)
        if split_name in split_writers:
            split_writers[split_name].write(row)
        return
    for split_name, ids in split_ids.items():
        if entity_id in ids and split_name in split_writers:
            split_writers[split_name].write(row)
            break


def _close_lmdb_writers(dataset_writer: TokenizedLmdbWriter | None, split_writers: dict[str, TokenizedLmdbWriter]) -> None:
    if dataset_writer is not None:
        dataset_writer.close()
    for writer in split_writers.values():
        writer.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--split_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_events", type=int, default=512)
    parser.add_argument("--max_event_tokens", type=int, default=24)
    parser.add_argument("--max_profile_tokens", type=int, default=200)
    parser.add_argument("--backend", choices=["parquet", "lmdb", "both"], default="parquet")
    parser.add_argument("--row_group_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lmdb_map_size_gb", type=int, default=64)
    parser.add_argument("--lmdb_commit_interval", type=int, default=1024)
    parser.add_argument("--hash_split_seed", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--valid_frac", type=float, default=0.1)

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
    split_dir = Path(args.split_dir) if args.split_dir else None
    split_ids = _load_split_ids(split_dir)
    use_hash_split = args.hash_split_seed is not None
    dataset_writer, split_writers = _build_lmdb_writers(
        out_dir=out_dir,
        backend=args.backend,
        split_ids=split_ids,
        use_hash_split=use_hash_split,
        map_size_gb=args.lmdb_map_size_gb,
        commit_interval=args.lmdb_commit_interval,
    )

    rows: list[dict[str, Any]] | None = [] if args.backend in {"parquet", "both"} else None
    num_workers = max(int(args.num_workers), 1)
    try:
        if num_workers == 1 or len(payloads) <= 1:
            _init_worker(vocab, cfg)
            iterator = (_encode_entity_payload(payload) for payload in payloads)
            for row in tqdm(iterator, desc="encode", total=len(payloads)):
                if rows is not None:
                    rows.append(row)
                _route_row_to_lmdb(
                    row,
                    dataset_writer,
                    split_writers,
                    split_ids,
                    args.hash_split_seed,
                    args.train_frac,
                    args.valid_frac,
                )
        else:
            mp_context = _resolve_mp_context(num_workers)
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(vocab, cfg),
            ) as executor:
                iterator = executor.map(_encode_entity_payload, payloads, chunksize=32)
                for row in tqdm(iterator, desc="encode", total=len(payloads)):
                    if rows is not None:
                        rows.append(row)
                    _route_row_to_lmdb(
                        row,
                        dataset_writer,
                        split_writers,
                        split_ids,
                        args.hash_split_seed,
                        args.train_frac,
                        args.valid_frac,
                    )
    finally:
        _close_lmdb_writers(dataset_writer, split_writers)

    row_group_size = max(int(args.row_group_size), 1)
    if rows is not None:
        df = pd.DataFrame(rows)
        _write_parquet(df, out_dir / "dataset.parquet", row_group_size=row_group_size)
        if split_dir:
            _write_split_parquets(df, split_dir, out_dir, row_group_size=row_group_size)
    num_records = len(rows) if rows is not None else len(payloads)
    write_json(
        out_dir / "tokenized_summary.json",
        {
            "num_records": int(num_records),
            "vocab_size": int(len(vocab.token_to_id)),
            "max_profile_tokens": int(args.max_profile_tokens),
            "max_event_tokens": int(args.max_event_tokens),
            "max_events": int(args.max_events),
            "row_group_size": row_group_size,
            "backend": args.backend,
            "hash_split_seed": args.hash_split_seed,
            "train_frac": float(args.train_frac),
            "valid_frac": float(args.valid_frac),
        },
    )


if __name__ == "__main__":
    main()
