from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.tokenizer.text_bpe import SimpleTextTokenizer
from src.tokenizer.time import periodic_encode, soft_log_seconds
from src.tokenizer.vocab import TokenizerVocab


@dataclass(frozen=True)
class StructuredRecordConfig:
    max_events: int = 256
    max_event_tokens: int = 24
    max_profile_tokens: int = 200
    history_time_anchor: Literal["evaluation", "last_event", "decoupled"] = "last_event"
    inactivity_profile_col: str = "seconds_since_last_event"


def calendar_features(ts: pd.Timestamp) -> list[float]:
    hour = periodic_encode(np.array([ts.hour], dtype=np.float64), period=24.0)[0]
    dow = periodic_encode(np.array([ts.dayofweek], dtype=np.float64), period=7.0)[0]
    dom = periodic_encode(np.array([ts.day], dtype=np.float64), period=31.0)[0]
    feats = np.concatenate([hour, dow, dom], axis=0)
    return feats.astype(np.float32).tolist()


def relative_time_feature(event_ts: pd.Timestamp, evaluation_ts: pd.Timestamp) -> float:
    try:
        delta_seconds = max(0.0, (evaluation_ts - event_ts).total_seconds())
    except Exception:
        return 0.0
    return float(soft_log_seconds(delta_seconds))


def _time_feature_to_anchor(event_ts: pd.Timestamp, anchor_ts: pd.Timestamp) -> float:
    try:
        delta_seconds = max(0.0, (anchor_ts - event_ts).total_seconds())
    except Exception:
        return 0.0
    return float(soft_log_seconds(delta_seconds))


def _is_time_like_column(col: str) -> bool:
    normalized = col.strip().lower()
    return any(
        token in normalized
        for token in ("time", "timestamp", "date", "seen")
    )


def _maybe_parse_ts(value: Any) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.tz_localize("UTC") if value.tzinfo is None else value
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _normalize_categorical_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "[NA]"
    return str(value).strip().lower() or "[NA]"


def _encode_numeric_value(vocab: TokenizerVocab, namespace: str, col: str, value: Any) -> int:
    field_key = f"{namespace}:{col}"
    if value is None or pd.isna(value):
        return vocab.encode_token(f"V:{field_key}#[NA]")
    try:
        numeric_value = float(value)
    except Exception:
        return vocab.encode_token(f"V:{field_key}#[NA]")
    if vocab.numeric_zero_bucket and numeric_value == 0.0:
        return vocab.encode_token(f"V:{field_key}#ZERO")
    bucket = vocab.numeric_binners.get(field_key, None)
    if bucket is None:
        return vocab.unk_id
    return vocab.encode_token(f"V:{field_key}#B{max(bucket.bucket(numeric_value), 0)}")


def _encode_categorical_value(vocab: TokenizerVocab, namespace: str, col: str, value: Any) -> int:
    field_key = f"{namespace}:{col}"
    token_value = _normalize_categorical_value(value)
    token = f"V:{field_key}={token_value}"
    if token in vocab.token_to_id:
        return vocab.token_to_id[token]
    return vocab.encode_token(f"V:{field_key}=[UNK]")


def _encode_textual_value(
    vocab: TokenizerVocab,
    namespace: str,
    col: str,
    value: Any,
    max_value_tokens: int,
) -> list[int]:
    _ = namespace, col
    if value is None or pd.isna(value):
        return [vocab.encode_token("T:[NA]")]
    text = str(value)
    if not text.strip():
        return [vocab.encode_token("T:[NA]")]
    tokenizer = vocab.text_tokenizer or SimpleTextTokenizer(vocab=["[UNK]"])
    encoding = tokenizer.encode(text)
    pieces = list(getattr(encoding, "tokens", []))[:max_value_tokens]
    if not pieces:
        return [vocab.encode_token("T:[UNK]")]
    return [vocab.encode_token(f"T:{piece}") for piece in pieces]


def _encode_field(
    vocab: TokenizerVocab,
    namespace: str,
    col: str,
    value: Any,
    max_value_tokens: int,
) -> list[int]:
    field_key = f"{namespace}:{col}"
    value_type = vocab.field_value_types.get(
        field_key,
        "numeric" if field_key in vocab.numeric_binners else "categorical",
    )
    if value_type == "numeric":
        return [_encode_numeric_value(vocab, namespace, col, value)]
    if value_type == "textual":
        return _encode_textual_value(vocab, namespace, col, value, max_value_tokens=max_value_tokens)
    return [_encode_categorical_value(vocab, namespace, col, value)]


def _collect_valid_event_rows(
    events: pd.DataFrame | list[dict[str, Any]],
    max_events: int,
) -> list[tuple[dict[str, Any] | pd.Series, pd.Timestamp]]:
    if isinstance(events, pd.DataFrame):
        event_rows = events.to_dict("records")
    else:
        event_rows = list(events)

    valid_rows: list[tuple[dict[str, Any] | pd.Series, pd.Timestamp]] = []
    for row in event_rows:
        if isinstance(row, pd.Series):
            timestamp = row["timestamp"]
        elif isinstance(row, dict) and "fields" in row:
            timestamp = row.get("timestamp")
        else:
            timestamp = row.get("timestamp") if isinstance(row, dict) else None
        ts = _maybe_parse_ts(timestamp)
        if ts is None:
            continue
        valid_rows.append((row, ts))
    if max_events > 0 and len(valid_rows) > max_events:
        valid_rows = valid_rows[-max_events:]
    return valid_rows


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
        value = profile[col] if col in profile else None
        key_id = vocab.encode_token(f"K:P:{col}")
        field_value_ids = _encode_field(
            vocab=vocab,
            namespace="P",
            col=col,
            value=value,
            max_value_tokens=vocab.max_value_tokens_per_field,
        )
        parsed_ts = _maybe_parse_ts(value) if _is_time_like_column(col) else None
        time_value = relative_time_feature(parsed_ts, evaluation_time) if parsed_ts is not None else 0.0
        for pos, value_id in enumerate(field_value_ids):
            if len(key_ids) >= max_profile_tokens:
                break
            key_ids.append(key_id)
            value_ids.append(value_id)
            value_pos.append(pos)
            profile_time.append(time_value)
        if len(key_ids) >= max_profile_tokens:
            break

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
    return _encode_event_features_with_anchor(
        vocab=vocab,
        events=events,
        evaluation_time=evaluation_time,
        max_events=max_events,
        max_event_tokens=max_event_tokens,
        history_time_anchor="evaluation",
    )


def _encode_event_features_with_anchor(
    vocab: TokenizerVocab,
    events: pd.DataFrame | list[dict[str, Any]],
    evaluation_time: pd.Timestamp,
    max_events: int,
    max_event_tokens: int,
    history_time_anchor: Literal["evaluation", "last_event", "decoupled"],
) -> dict[str, list]:
    included_rows = _collect_valid_event_rows(events, max_events=max_events)
    if included_rows:
        last_event_ts = max(ts for _, ts in included_rows)
        inactivity_gap_seconds = max(0.0, (evaluation_time - last_event_ts).total_seconds())
    else:
        last_event_ts = evaluation_time
        inactivity_gap_seconds = 0.0

    event_key_ids: list[list[int]] = []
    event_value_ids: list[list[int]] = []
    event_value_pos: list[list[int]] = []
    event_token_mask: list[list[int]] = []
    event_time: list[float] = []
    calendar_rows: list[list[float]] = []
    event_mask: list[int] = []

    for row, ts in included_rows:
        if isinstance(row, pd.Series):
            fields = row
        elif isinstance(row, dict) and "fields" in row:
            fields = row.get("fields", {}) or {}
        else:
            fields = row

        key_ids: list[int] = []
        value_ids: list[int] = []
        value_pos: list[int] = []
        for col in vocab.event_cols:
            if len(key_ids) >= max_event_tokens:
                break
            value = fields[col] if col in fields else None
            key_id = vocab.encode_token(f"K:E:{col}")
            field_value_ids = _encode_field(
                vocab=vocab,
                namespace="E",
                col=col,
                value=value,
                max_value_tokens=vocab.max_value_tokens_per_field,
            )
            remaining = max_event_tokens - len(key_ids)
            for pos, value_id in enumerate(field_value_ids[:remaining]):
                key_ids.append(key_id)
                value_ids.append(value_id)
                value_pos.append(pos)

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
        if history_time_anchor == "evaluation":
            anchor_ts = evaluation_time
        elif history_time_anchor in {"last_event", "decoupled"}:
            anchor_ts = last_event_ts
        else:
            raise ValueError(f"Unknown history_time_anchor: {history_time_anchor}")
        event_time.append(_time_feature_to_anchor(ts, anchor_ts))
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
        "seconds_since_last_event": float(inactivity_gap_seconds),
        "last_event_ts": last_event_ts.isoformat() if last_event_ts is not None else "",
    }


def encode_record(
    vocab: TokenizerVocab,
    profile: pd.Series | dict[str, Any],
    events: pd.DataFrame | list[dict[str, Any]],
    evaluation_time: pd.Timestamp,
    cfg: StructuredRecordConfig,
) -> dict[str, list]:
    event_encoded = _encode_event_features_with_anchor(
        vocab=vocab,
        events=events,
        evaluation_time=evaluation_time,
        max_events=cfg.max_events,
        max_event_tokens=cfg.max_event_tokens,
        history_time_anchor=cfg.history_time_anchor,
    )

    if isinstance(profile, pd.Series):
        profile_aug: dict[str, Any] = profile.to_dict()
    else:
        profile_aug = dict(profile)
    if cfg.history_time_anchor == "decoupled":
        profile_aug[cfg.inactivity_profile_col] = float(event_encoded["seconds_since_last_event"])

    profile_encoded = encode_profile_features(
        vocab=vocab,
        profile=profile_aug,
        evaluation_time=evaluation_time,
        max_profile_tokens=cfg.max_profile_tokens,
    )

    encoded = {}
    encoded.update(profile_encoded)
    encoded.update(event_encoded)
    encoded.pop("last_event_ts", None)
    encoded.pop("seconds_since_last_event", None)
    return encoded
