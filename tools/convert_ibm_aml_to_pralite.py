from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json


def _normalize_key(value: object) -> str:
    if pd.isna(value):
        return "[NA]"
    return str(value).strip()


def _pick_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise ValueError(f"None of these columns found: {candidates}; available={columns}")
    return None


def _resolve_raw_csv(raw_dir: Path, raw_csv: str | None) -> Path:
    if raw_csv:
        path = Path(raw_csv)
        if path.is_absolute() and path.exists():
            return path
        if path.exists():
            return path.resolve()
        candidate = raw_dir / raw_csv
        if candidate.exists():
            return candidate
        path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Missing IBM AML raw CSV: {path}")
        return path

    preferred = [
        raw_dir / "LI-Small_Trans.csv",
        raw_dir / "LI-Medium_Trans.csv",
        raw_dir / "LI-Large_Trans.csv",
        raw_dir / "HI-Small_Trans.csv",
        raw_dir / "HI-Medium_Trans.csv",
        raw_dir / "HI-Large_Trans.csv",
        raw_dir / "transactions.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(raw_dir.rglob("*_Trans.csv")) + sorted(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find IBM AML transactions CSV under {raw_dir}")
    if len(candidates) == 1:
        return candidates[0]

    sample = "\n".join(f"- {path}" for path in candidates[:10])
    raise FileNotFoundError(
        "Found multiple IBM AML CSV files. Set --raw_csv explicitly. Candidates:\n" + sample
    )


def _resolve_accounts_csv(raw_dir: Path, tx_csv: Path) -> Path | None:
    if tx_csv.name.endswith("_Trans.csv"):
        stem = tx_csv.name.replace("_Trans.csv", "_accounts.csv")
        direct = tx_csv.parent / stem
        if direct.exists() and direct != tx_csv:
            return direct
    candidates = sorted(raw_dir.rglob("*_accounts.csv"))
    return candidates[0] if candidates else None


def _parse_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.notna().any():
        return parsed
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(numeric.fillna(0), unit="h")
    raise ValueError("Could not parse IBM AML timestamp column")


def _stable_entity_id(account_key: str) -> int:
    digest = hashlib.blake2b(account_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def _read_transactions(
    csv_path: Path,
    usecols: list[str],
    chunksize: int,
    sample_frac: float,
    seed: int,
) -> pd.DataFrame:
    if chunksize <= 0:
        return pd.read_csv(csv_path, usecols=usecols)

    frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize), start=1):
        total_rows += len(chunk)
        if 0.0 < sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=seed + chunk_idx)
        kept_rows += len(chunk)
        frames.append(chunk)
        print(
            f"[convert_ibm_aml] chunk={chunk_idx} raw_rows={total_rows} kept_rows={kept_rows}",
            flush=True,
        )
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def convert_ibm_aml_to_pralite(
    raw_dir: str | Path,
    processed_dir: str | Path,
    raw_csv: str | None = None,
    chunksize: int = 0,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> Path:
    raw_path = Path(raw_dir)
    out_dir = ensure_dir(processed_dir)
    csv_path = _resolve_raw_csv(raw_path, raw_csv)
    accounts_csv = _resolve_accounts_csv(raw_path, csv_path)

    header = pd.read_csv(csv_path, nrows=0)
    columns = [str(col) for col in header.columns]

    time_col = _pick_column(columns, ["Timestamp", "timestamp", "time", "datetime", "Date"])
    from_bank_col = _pick_column(columns, ["From Bank", "sender_bank", "source_bank", "bankOrig", "bank_from"])
    from_account_col = _pick_column(
        columns,
        ["From Account", "Account", "sender_account", "source_account", "nameOrig", "account_from"],
    )
    to_bank_col = _pick_column(columns, ["To Bank", "receiver_bank", "dest_bank", "bankDest", "bank_to"])
    to_account_col = _pick_column(
        columns,
        ["To Account", "Account.1", "receiver_account", "dest_account", "nameDest", "account_to"],
    )
    amount_paid_col = _pick_column(columns, ["Amount Paid", "amount_paid", "amount", "Amount"], required=False)
    amount_received_col = _pick_column(columns, ["Amount Received", "amount_received"], required=False)
    payment_currency_col = _pick_column(columns, ["Payment Currency", "payment_currency", "currency"], required=False)
    receiving_currency_col = _pick_column(columns, ["Receiving Currency", "receiving_currency"], required=False)
    payment_format_col = _pick_column(columns, ["Payment Format", "payment_format", "type"], required=False)
    laundering_col = _pick_column(columns, ["Is Laundering", "is_laundering", "label", "target"])
    tx_id_col = _pick_column(columns, ["transaction_id", "Transaction ID", "tx_id", "id"], required=False)

    usecols = [
        col
        for col in [
            time_col,
            from_bank_col,
            from_account_col,
            to_bank_col,
            to_account_col,
            amount_paid_col,
            amount_received_col,
            payment_currency_col,
            receiving_currency_col,
            payment_format_col,
            laundering_col,
            tx_id_col,
        ]
        if col is not None
    ]
    read_started_at = time.perf_counter()
    tx = _read_transactions(csv_path, usecols=usecols, chunksize=max(int(chunksize), 0), sample_frac=float(sample_frac), seed=int(seed))
    print(
        f"[convert_ibm_aml] loaded rows={len(tx)} elapsed_s={time.perf_counter() - read_started_at:.2f}",
        flush=True,
    )

    tx["timestamp"] = _parse_timestamp(tx[time_col])
    tx = tx.dropna(subset=["timestamp"]).reset_index(drop=True)

    sender_key = tx[from_bank_col].map(_normalize_key) + "::" + tx[from_account_col].map(_normalize_key)
    receiver_key = tx[to_bank_col].map(_normalize_key) + "::" + tx[to_account_col].map(_normalize_key)
    tx["sender_entity_id"] = sender_key.map(_stable_entity_id).astype("int64")
    tx["receiver_entity_id"] = receiver_key.map(_stable_entity_id).astype("int64")
    tx["event_id"] = (
        pd.to_numeric(tx[tx_id_col], errors="coerce").fillna(pd.Series(np.arange(len(tx)), index=tx.index)).astype("int64")
        if tx_id_col is not None
        else pd.Series(np.arange(len(tx), dtype=np.int64), index=tx.index)
    )
    tx["label"] = pd.to_numeric(tx[laundering_col], errors="coerce").fillna(0).astype("int64")
    tx["amount_paid"] = pd.to_numeric(tx[amount_paid_col], errors="coerce") if amount_paid_col else 0.0
    tx["amount_received"] = pd.to_numeric(tx[amount_received_col], errors="coerce") if amount_received_col else tx["amount_paid"]
    tx["payment_currency"] = (
        tx[payment_currency_col].fillna("UNK").astype(str) if payment_currency_col else "UNK"
    )
    tx["receiving_currency"] = (
        tx[receiving_currency_col].fillna("UNK").astype(str) if receiving_currency_col else tx["payment_currency"]
    )
    tx["payment_format"] = tx[payment_format_col].fillna("UNK").astype(str) if payment_format_col else "UNK"
    tx["sender_bank"] = tx[from_bank_col].map(_normalize_key)
    tx["receiver_bank"] = tx[to_bank_col].map(_normalize_key)
    tx["sender_account"] = tx[from_account_col].map(_normalize_key)
    tx["receiver_account"] = tx[to_account_col].map(_normalize_key)

    src_events = pd.DataFrame(
        {
            "entity_id": tx["sender_entity_id"].astype("int64"),
            "event_id": (tx["event_id"] * 2).astype("int64"),
            "timestamp": tx["timestamp"],
            "event_type": "aml_transaction",
            "direction": "out",
            "counterparty_id": tx["receiver_entity_id"].astype("int64"),
            "bank_id": tx["sender_bank"],
            "counterparty_bank": tx["receiver_bank"],
            "amount_paid": tx["amount_paid"],
            "amount_received": tx["amount_received"],
            "payment_currency": tx["payment_currency"],
            "receiving_currency": tx["receiving_currency"],
            "payment_format": tx["payment_format"],
            "label": tx["label"],
        }
    )
    dst_events = pd.DataFrame(
        {
            "entity_id": tx["receiver_entity_id"].astype("int64"),
            "event_id": (tx["event_id"] * 2 + 1).astype("int64"),
            "timestamp": tx["timestamp"],
            "event_type": "aml_transaction",
            "direction": "in",
            "counterparty_id": tx["sender_entity_id"].astype("int64"),
            "bank_id": tx["receiver_bank"],
            "counterparty_bank": tx["sender_bank"],
            "amount_paid": tx["amount_received"],
            "amount_received": tx["amount_paid"],
            "payment_currency": tx["receiving_currency"],
            "receiving_currency": tx["payment_currency"],
            "payment_format": tx["payment_format"],
            "label": tx["label"],
        }
    )

    events = pd.concat([src_events, dst_events], ignore_index=True)
    events = events.sort_values(["entity_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)

    grouped = events.groupby("entity_id", sort=False)
    direction_counts = events.pivot_table(
        index="entity_id",
        columns="direction",
        values="event_id",
        aggfunc="count",
        fill_value=0,
    )
    profiles = grouped.agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        tx_count=("event_id", "count"),
        mean_amount=("amount_paid", "mean"),
        primary_bank=("bank_id", "first"),
        dominant_payment_format=(
            "payment_format",
            lambda s: s.mode().iloc[0] if not s.mode().empty else "UNK",
        ),
    ).reset_index()
    profiles["sent_tx_count"] = profiles["entity_id"].map(direction_counts.get("out", pd.Series(dtype="int64"))).fillna(0).astype("int64")
    profiles["recv_tx_count"] = profiles["entity_id"].map(direction_counts.get("in", pd.Series(dtype="int64"))).fillna(0).astype("int64")
    profiles["bank_name"] = "unknown"
    profiles["source_entity_id"] = "unknown"
    profiles["entity_name"] = "unknown"

    if accounts_csv is not None and accounts_csv.exists():
        accounts = pd.read_csv(accounts_csv)
        account_columns = [str(col) for col in accounts.columns]
        bank_id_col = _pick_column(account_columns, ["Bank ID", "bank_id", "Bank"], required=False)
        account_number_col = _pick_column(account_columns, ["Account Number", "account_number", "Account"], required=False)
        entity_id_col = _pick_column(account_columns, ["Entity ID", "entity_id"], required=False)
        entity_name_col = _pick_column(account_columns, ["Entity Name", "entity_name"], required=False)
        bank_name_col = _pick_column(account_columns, ["Bank Name", "bank_name"], required=False)
        if bank_id_col and account_number_col:
            lookup = accounts.copy()
            lookup["account_key"] = (
                lookup[bank_id_col].map(_normalize_key) + "::" + lookup[account_number_col].map(_normalize_key)
            )
            sender_lookup = pd.DataFrame(
                {
                    "entity_id": tx["sender_entity_id"].astype("int64"),
                    "account_key": tx["sender_bank"] + "::" + tx["sender_account"],
                }
            )
            receiver_lookup = pd.DataFrame(
                {
                    "entity_id": tx["receiver_entity_id"].astype("int64"),
                    "account_key": tx["receiver_bank"] + "::" + tx["receiver_account"],
                }
            )
            entity_lookup = pd.concat([sender_lookup, receiver_lookup], ignore_index=True).drop_duplicates("entity_id")
            entity_lookup = entity_lookup.merge(
                lookup[
                    [
                        "account_key",
                        *([entity_id_col] if entity_id_col else []),
                        *([entity_name_col] if entity_name_col else []),
                        *([bank_name_col] if bank_name_col else []),
                    ]
                ],
                on="account_key",
                how="left",
            )
            if entity_id_col:
                profiles["source_entity_id"] = profiles["entity_id"].map(
                    entity_lookup.set_index("entity_id")[entity_id_col].fillna("unknown")
                )
            if entity_name_col:
                profiles["entity_name"] = profiles["entity_id"].map(
                    entity_lookup.set_index("entity_id")[entity_name_col].fillna("unknown")
                )
            if bank_name_col:
                profiles["bank_name"] = profiles["entity_id"].map(
                    entity_lookup.set_index("entity_id")[bank_name_col].fillna("unknown")
                )

    labels = grouped["label"].max().reset_index().rename(columns={"label": "label"})
    evaluation_time = grouped["timestamp"].max().reset_index().rename(columns={"timestamp": "evaluation_time"})
    labels = labels.merge(evaluation_time, on="entity_id", how="left")
    labels["evaluation_time"] = labels["evaluation_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    events = events.drop(columns=["label"])
    profiles.to_parquet(out_dir / "profiles.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    write_json(
        out_dir / "schema.json",
        {
            "dataset": "ibm_aml",
            "processed_dir": str(out_dir),
            "source_csv": str(csv_path),
            "accounts_csv": str(accounts_csv) if accounts_csv is not None else None,
            "profile_columns": [c for c in profiles.columns if c != "entity_id"],
            "event_columns": [c for c in events.columns if c not in {"entity_id", "event_id"}],
            "label_column": "label",
            "evaluation_time_column": "evaluation_time",
            "notes": "IBM AML adaptation builds account-level histories with both outgoing and incoming transaction events.",
        },
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/ibm_aml")
    parser.add_argument("--processed_dir", default="data/processed/ibm_aml_full")
    parser.add_argument("--raw_csv", default=None)
    parser.add_argument("--chunksize", type=int, default=0)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert_ibm_aml_to_pralite(
        args.raw_dir,
        args.processed_dir,
        raw_csv=args.raw_csv,
        chunksize=args.chunksize,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
