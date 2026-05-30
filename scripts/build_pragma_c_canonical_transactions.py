#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.fs import ensure_dir, write_json
from tools.convert_ibm_aml_to_pralite import (
    _normalize_key,
    _parse_timestamp,
    _pick_column,
    _resolve_raw_csv,
    _stable_entity_id,
)


def build_canonical_transactions(
    raw_dir: str | Path,
    output_root: str | Path,
    *,
    raw_csv: str | None = None,
) -> Path:
    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    canonical_dir = ensure_dir(output_root / "canonical")
    raw_meta_dir = ensure_dir(output_root / "raw_metadata")

    csv_path = _resolve_raw_csv(raw_dir, raw_csv)
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

    tx = pd.read_csv(csv_path)
    tx["transaction_time"] = _parse_timestamp(tx[time_col])
    tx = tx.dropna(subset=["transaction_time"]).reset_index(drop=True)

    tx["from_bank"] = tx[from_bank_col].map(_normalize_key)
    tx["from_account"] = tx[from_account_col].map(_normalize_key)
    tx["to_bank"] = tx[to_bank_col].map(_normalize_key)
    tx["to_account"] = tx[to_account_col].map(_normalize_key)

    sender_key = tx["from_bank"] + "::" + tx["from_account"]
    receiver_key = tx["to_bank"] + "::" + tx["to_account"]
    tx["sender_entity_id"] = sender_key.map(_stable_entity_id).astype("int64")
    tx["receiver_entity_id"] = receiver_key.map(_stable_entity_id).astype("int64")

    tx["amount_paid"] = pd.to_numeric(tx[amount_paid_col], errors="coerce").fillna(0.0) if amount_paid_col else 0.0
    tx["amount_received"] = (
        pd.to_numeric(tx[amount_received_col], errors="coerce").fillna(tx["amount_paid"])
        if amount_received_col
        else tx["amount_paid"]
    )
    tx["payment_currency"] = tx[payment_currency_col].astype("string").fillna("UNK") if payment_currency_col else "UNK"
    tx["receiving_currency"] = (
        tx[receiving_currency_col].astype("string").fillna("UNK")
        if receiving_currency_col
        else tx["payment_currency"]
    )
    tx["payment_format"] = tx[payment_format_col].astype("string").fillna("UNK") if payment_format_col else "UNK"
    tx["is_laundering"] = pd.to_numeric(tx[laundering_col], errors="coerce").fillna(0).astype("int64")
    tx["source_file"] = csv_path.name
    tx["source_row"] = np.arange(len(tx), dtype=np.int64)

    canonical = tx[
        [
            "transaction_time",
            "from_bank",
            "from_account",
            "to_bank",
            "to_account",
            "sender_entity_id",
            "receiver_entity_id",
            "amount_received",
            "receiving_currency",
            "amount_paid",
            "payment_currency",
            "payment_format",
            "is_laundering",
            "source_file",
            "source_row",
        ]
    ].copy()
    canonical = canonical.sort_values(["transaction_time", "source_row"], kind="stable").reset_index(drop=True)
    canonical.insert(0, "transaction_id", np.arange(len(canonical), dtype=np.int64))
    canonical.to_parquet(canonical_dir / "transactions.parquet", index=False)

    write_json(
        raw_meta_dir / "schema.json",
        {
            "dataset_name": "ibm_aml_li_medium_pragma_c",
            "source_csv": str(csv_path.resolve()),
            "canonical_columns": canonical.columns.tolist(),
            "time_column": "transaction_time",
            "label_column": "is_laundering",
        },
    )
    write_json(
        raw_meta_dir / "source_files.json",
        {
            "files": [
                {
                    "path": str(csv_path.resolve()),
                    "rows": int(len(canonical)),
                }
            ]
        },
    )
    return canonical_dir / "transactions.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/ibm_aml")
    parser.add_argument("--raw_csv", default=None)
    parser.add_argument("--output_root", default="data/streaming/ibm_aml_li_medium_pragma_c")
    args = parser.parse_args()
    build_canonical_transactions(args.raw_dir, args.output_root, raw_csv=args.raw_csv)


if __name__ == "__main__":
    main()
