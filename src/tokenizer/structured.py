from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.tokenizer.time import periodic_encode, soft_log_seconds
from src.tokenizer.vocab import TokenizerVocab


@dataclass(frozen=True)
class StructuredRecordConfig:
    max_events: int = 512
    max_event_tokens: int = 24
    max_profile_tokens: int = 200


def calendar_features(ts: pd.Timestamp) -> list[float]:
    hour = periodic_encode(np.array([ts.hour], dtype=np.float64), period=24.0)[0]
    dow = periodic_encode(np.array([ts.dayofweek], dtype=np.float64), period=7.0)[0]
    dom = periodic_encode(np.array([ts.day], dtype=np.float64), period=31.0)[0]
    feats = np.concatenate([hour, dow, dom], axis=0)
    return feats.astype(np.float32).tolist()


def relative_time_feature(event_ts: pd.Timestamp, evaluation_ts: pd.Timestamp) -> float:
    delta_seconds = max(0.0, (evaluation_ts - event_ts).total_seconds())
    return float(soft_log_seconds(delta_seconds))


def _maybe_parse_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _encode_value(vocab: TokenizerVocab, namespace: str, col: str, value: Any) -> int:
    col_key = f"{namespace}:{col}"
    if col_key in vocab.numeric_binners:
        bucket = vocab.numeric_binners[col_key].bucket(value)
        bucket = max(bucket, 0)
        return vocab.encode_token(f"V:{namespace}:{col}#B{bucket}")
    token_value = "[NA]" if pd.isna(value) else str(value)
    return vocab.encode_token(f"V:{namespace}:{col}={token_value}")


def _pad_pairs(
    key_ids: list[int],
    value_ids: list[int],
    value_pos: list[int],
    time_values: list[float],
    length: int,
    pad_id: int,
) -> tuple[list[int], list[int], list[int], list[float], list[int]]:
    key_ids = key_ids[:length]
    value_ids = value_ids[:length]
    value_pos = value_pos[:length]
    time_values = time_values[:length]
    mask = [1] * len(key_ids) + [0] * max(length - len(key_ids), 0)
    pad_len = max(length - len(key_ids), 0)
    return (
        key_ids + [pad_id] * pad_len,
        value_ids + [pad_id] * pad_len,
        value_pos + [0] * pad_len,
        time_values + [0.0] * pad_len,
        mask,
    )


def encode_profile_features(
    vocab: TokenizerVocab,
    profile: pd.Series | dict[str, Any],
    evaluation_time: pd.Timestamp,
    max_profile_tokens: int,
) -> dict[str, list[int] | list[float]]:
    key_ids: list[int] = []
    value_ids: list[int] = []
    value_pos: list[int] = []
    profile_time: list[float] = []

    for col in vocab.profile_cols:
        key_ids.append(vocab.encode_token(f"K:P:{col}"))
        value = profile[col] if col in profile else None
        value_ids.append(_encode_value(vocab, "P", col, value))
        value_pos.append(0)

        parsed_ts = _maybe_parse_ts(value)
        if parsed_ts is not None:
            profile_time.append(relative_time_feature(parsed_ts, evaluation_time))
        else:
            profile_time.append(0.0)

    key_ids, value_ids, value_pos, profile_time, profile_mask = _pad_pairs(
        key_ids,
        value_ids,
        value_pos,
        profile_time,
        length=max_profile_tokens,
        pad_id=vocab.pad_id,
    )
    return {
        "profile_key_ids": key_ids,
        "profile_value_ids": value_ids,
        "profile_value_pos": value_pos,
        "profile_time": profile_time,
        "profile_mask": profile_mask,
    }


def encode_event_features(
    vocab: TokenizerVocab,
    events: pd.DataFrame | list[dict[str, Any]],
    evaluation_time: pd.Timestamp,
    max_events: int,
    max_event_tokens: int,
) -> dict[str, list]:
    if isinstance(events, pd.DataFrame):
        event_rows = [row for _, row in events.iterrows()]
    else:
        event_rows = list(events)

    event_key_ids: list[list[int]] = []
    event_value_ids: list[list[int]] = []
    event_value_pos: list[list[int]] = []
    event_token_mask: list[list[int]] = []
    event_time: list[float] = []
    calendar_rows: list[list[float]] = []
    event_mask: list[int] = []

    for row in event_rows[:max_events]:
        if isinstance(row, pd.Series):
            fields = row
            timestamp = row["timestamp"]
        else:
            fields = row.get("fields", {}) or {}
            timestamp = row.get("timestamp")
        ts = _maybe_parse_ts(timestamp)
        if ts is None:
            continue

        key_ids: list[int] = []
        value_ids: list[int] = []
        value_pos: list[int] = []
        for col in vocab.event_cols:
            value = fields[col] if col in fields else None
            key_ids.append(vocab.encode_token(f"K:E:{col}"))
            value_ids.append(_encode_value(vocab, "E", col, value))
            value_pos.append(0)

        padded_key_ids, padded_value_ids, padded_value_pos, _, padded_mask = _pad_pairs(
            key_ids,
            value_ids,
            value_pos,
            [0.0] * len(key_ids),
            length=max_event_tokens,
            pad_id=vocab.pad_id,
        )
        event_key_ids.append(padded_key_ids)
        event_value_ids.append(padded_value_ids)
        event_value_pos.append(padded_value_pos)
        event_token_mask.append(padded_mask)
        event_time.append(relative_time_feature(ts, evaluation_time))
        calendar_rows.append(calendar_features(ts))
        event_mask.append(1)

    while len(event_key_ids) < max_events:
        event_key_ids.append([vocab.pad_id] * max_event_tokens)
        event_value_ids.append([vocab.pad_id] * max_event_tokens)
        event_value_pos.append([0] * max_event_tokens)
        event_token_mask.append([0] * max_event_tokens)
        event_time.append(0.0)
        calendar_rows.append([0.0] * 6)
        event_mask.append(0)

    return {
        "event_key_ids": event_key_ids,
        "event_value_ids": event_value_ids,
        "event_value_pos": event_value_pos,
        "event_token_mask": event_token_mask,
        "event_time": event_time,
        "calendar_features": calendar_rows,
        "event_mask": event_mask,
    }


def encode_record(
    vocab: TokenizerVocab,
    profile: pd.Series | dict[str, Any],
    events: pd.DataFrame | list[dict[str, Any]],
    evaluation_time: pd.Timestamp,
    cfg: StructuredRecordConfig,
) -> dict[str, list]:
    encoded = {}
    encoded.update(
        encode_profile_features(
            vocab=vocab,
            profile=profile,
            evaluation_time=evaluation_time,
            max_profile_tokens=cfg.max_profile_tokens,
        )
    )
    encoded.update(
        encode_event_features(
            vocab=vocab,
            events=events,
            evaluation_time=evaluation_time,
            max_events=cfg.max_events,
            max_event_tokens=cfg.max_event_tokens,
        )
    )
    return encoded
