from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

PROFILE_COLS = [
    "account_age_days",
    "tx_count_so_far",
    "sent_tx_count_so_far",
    "recv_tx_count_so_far",
    "total_out_amount_so_far",
    "total_in_amount_so_far",
    "unique_counterparties_so_far",
    "recent_24h_txn_count",
    "recent_7d_txn_count",
    "primary_bank_so_far",
    "dominant_payment_format_so_far",
]

EVENT_COLS = [
    "event_type",
    "direction",
    "bank_id",
    "counterparty_bank",
    "amount_paid",
    "amount_received",
    "payment_currency",
    "receiving_currency",
    "payment_format",
]

STAGE_C_SPLITS = ("train", "valid", "calibration", "test", "embargo")
PRETRAIN_EVAL_SOURCES = {"transaction_sender", "transaction_receiver"}


def stable_hash_bucket(value: object, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % num_buckets


def build_account_event_view(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["transaction_time"] = pd.to_datetime(tx["transaction_time"], utc=True, errors="coerce")
    sender_rows = pd.DataFrame(
        {
            "entity_id": tx["sender_entity_id"].astype("int64"),
            "transaction_id": tx["transaction_id"].astype("int64"),
            "timestamp": tx["transaction_time"],
            "event_type": "aml_transaction",
            "direction": "out",
            "bank_id": tx["from_bank"].astype("string").fillna("UNK"),
            "counterparty_bank": tx["to_bank"].astype("string").fillna("UNK"),
            "amount_paid": pd.to_numeric(tx["amount_paid"], errors="coerce").fillna(0.0).astype("float64"),
            "amount_received": pd.to_numeric(tx["amount_received"], errors="coerce").fillna(0.0).astype("float64"),
            "payment_currency": tx["payment_currency"].astype("string").fillna("UNK"),
            "receiving_currency": tx["receiving_currency"].astype("string").fillna("UNK"),
            "payment_format": tx["payment_format"].astype("string").fillna("UNK"),
            "counterparty_id": tx["receiver_entity_id"].astype("int64"),
            "label": pd.to_numeric(tx["is_laundering"], errors="coerce").fillna(0).astype("int64"),
        }
    )
    receiver_rows = pd.DataFrame(
        {
            "entity_id": tx["receiver_entity_id"].astype("int64"),
            "transaction_id": tx["transaction_id"].astype("int64"),
            "timestamp": tx["transaction_time"],
            "event_type": "aml_transaction",
            "direction": "in",
            "bank_id": tx["to_bank"].astype("string").fillna("UNK"),
            "counterparty_bank": tx["from_bank"].astype("string").fillna("UNK"),
            "amount_paid": pd.to_numeric(tx["amount_received"], errors="coerce").fillna(0.0).astype("float64"),
            "amount_received": pd.to_numeric(tx["amount_paid"], errors="coerce").fillna(0.0).astype("float64"),
            "payment_currency": tx["receiving_currency"].astype("string").fillna("UNK"),
            "receiving_currency": tx["payment_currency"].astype("string").fillna("UNK"),
            "payment_format": tx["payment_format"].astype("string").fillna("UNK"),
            "counterparty_id": tx["sender_entity_id"].astype("int64"),
            "label": pd.to_numeric(tx["is_laundering"], errors="coerce").fillna(0).astype("int64"),
        }
    )
    events = pd.concat([sender_rows, receiver_rows], ignore_index=True)
    return events.sort_values(["entity_id", "timestamp", "transaction_id", "direction"], kind="stable").reset_index(drop=True)


def history_before(
    account_events: pd.DataFrame,
    evaluation_time: pd.Timestamp,
    *,
    max_history_events: int | None = None,
) -> pd.DataFrame:
    if account_events.empty:
        return account_events.iloc[0:0].copy()
    timestamps = account_events["timestamp"].to_numpy(dtype="datetime64[ns]")
    cutoff = np.searchsorted(timestamps, np.datetime64(evaluation_time.to_datetime64()), side="left")
    history = account_events.iloc[:cutoff]
    if max_history_events is not None and max_history_events > 0:
        history = history.tail(int(max_history_events))
    return history.copy()


def _series_mode(values: pd.Series, default: str = "UNK") -> str:
    if values.empty:
        return default
    mode = values.astype("string").fillna(default).mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def compute_profile_state(history: pd.DataFrame, evaluation_time: pd.Timestamp) -> dict[str, Any]:
    if history.empty:
        return {
            "account_age_days": 0.0,
            "tx_count_so_far": 0,
            "sent_tx_count_so_far": 0,
            "recv_tx_count_so_far": 0,
            "total_out_amount_so_far": 0.0,
            "total_in_amount_so_far": 0.0,
            "unique_counterparties_so_far": 0,
            "recent_24h_txn_count": 0,
            "recent_7d_txn_count": 0,
            "primary_bank_so_far": "UNK",
            "dominant_payment_format_so_far": "UNK",
        }
    timestamps = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    first_seen = timestamps.iloc[0]
    recent_24h = timestamps >= (evaluation_time - pd.Timedelta(hours=24))
    recent_7d = timestamps >= (evaluation_time - pd.Timedelta(days=7))
    sent_mask = history["direction"] == "out"
    recv_mask = history["direction"] == "in"
    return {
        "account_age_days": max((evaluation_time - first_seen).total_seconds() / 86400.0, 0.0),
        "tx_count_so_far": int(len(history)),
        "sent_tx_count_so_far": int(sent_mask.sum()),
        "recv_tx_count_so_far": int(recv_mask.sum()),
        "total_out_amount_so_far": float(pd.to_numeric(history.loc[sent_mask, "amount_paid"], errors="coerce").fillna(0.0).sum()),
        "total_in_amount_so_far": float(pd.to_numeric(history.loc[recv_mask, "amount_received"], errors="coerce").fillna(0.0).sum()),
        "unique_counterparties_so_far": int(history["counterparty_id"].nunique()),
        "recent_24h_txn_count": int(recent_24h.sum()),
        "recent_7d_txn_count": int(recent_7d.sum()),
        "primary_bank_so_far": _series_mode(history["bank_id"]),
        "dominant_payment_format_so_far": _series_mode(history["payment_format"]),
    }


def deterministic_time_cap(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(group) <= max_points:
        return group
    ordered = group.sort_values(["evaluation_time", "anchor_transaction_id", "eval_id"], kind="stable").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=max_points, dtype=int)
    return ordered.iloc[np.unique(positions)].copy()


def apply_split_caps(eval_points: pd.DataFrame, split_caps: dict[str, int]) -> pd.DataFrame:
    if eval_points.empty:
        return eval_points.copy()
    kept_groups: list[pd.DataFrame] = []
    for split_name, split_df in eval_points.groupby("split", sort=False):
        cap = int(split_caps.get(str(split_name), 0))
        if cap <= 0:
            kept_groups.append(split_df.copy())
            continue
        for _, entity_df in split_df.groupby("entity_id", sort=False):
            kept_groups.append(deterministic_time_cap(entity_df, cap))
    if not kept_groups:
        return eval_points.iloc[0:0].copy()
    return pd.concat(kept_groups, ignore_index=True).sort_values(
        ["split", "evaluation_time", "entity_id", "anchor_transaction_id"],
        kind="stable",
    )
