from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir


ACCOUNT_ID_OFFSET = 0
MERCHANT_ID_OFFSET = 10_000_000


@dataclass(frozen=True)
class PreparedTransxionPaths:
    raw_dir: Path
    out_dir: Path


def first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise ValueError(f"None of these columns found: {candidates}; available={list(df.columns)}")
    return None


def _normalize_key(value: object) -> str:
    if pd.isna(value):
        return "[NA]"
    return str(value).strip()


def _build_id_map(values: pd.Series, offset: int = 0) -> tuple[pd.Series, dict[str, int]]:
    normalized = values.map(_normalize_key)
    uniques = pd.Index(pd.unique(normalized))
    mapping = {key: offset + idx for idx, key in enumerate(uniques)}
    numeric = normalized.map(mapping).astype("int64")
    return numeric, mapping


def _pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return first_existing(df, candidates, required=False)


def _build_accounts(persons: pd.DataFrame) -> pd.DataFrame:
    region_col = _pick_optional_column(persons, ["region", "home_region", "country", "city"])
    created_col = _pick_optional_column(persons, ["created_at", "created_time", "signup_time", "register_time"])

    account_region = persons[region_col].astype("string") if region_col else pd.Series("unknown", index=persons.index)
    created_at = (
        pd.to_datetime(persons[created_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if created_col
        else pd.Series("1970-01-01T00:00:00Z", index=persons.index)
    )
    created_at = created_at.fillna("1970-01-01T00:00:00Z")

    return pd.DataFrame(
        {
            "account_id": persons["person_id"].astype("int64"),
            "person_id": persons["person_id"].astype("int64"),
            "entity_type": "person",
            "account_region": account_region.fillna("unknown").astype(str),
            "created_at": created_at.astype(str),
        }
    )


def prepare_transxion_public_raw(raw_dir: str | Path, out_dir: str | Path) -> PreparedTransxionPaths:
    raw_path = Path(raw_dir)
    out_path = ensure_dir(out_dir)

    person = pd.read_csv(raw_path / "person.csv")
    merchant = pd.read_csv(raw_path / "merchant.csv")
    tx = pd.read_csv(raw_path / "tx.csv")

    person_id_col = first_existing(person, ["person_id", "account_id", "id", "entity_id", "node_id"])
    merchant_id_col = first_existing(merchant, ["merchant_id", "id", "entity_id", "node_id"], required=False)

    tx_id_col = first_existing(tx, ["transaction_id", "tx_id", "id"], required=False)
    tx_sender_col = first_existing(tx, ["sender_id", "src_id", "source_id", "from_id", "payer_id", "orig_id", "nameorig"])
    tx_receiver_col = first_existing(
        tx,
        ["receiver_id", "dst_id", "target_id", "to_id", "payee_id", "dest_id", "namedest"],
        required=False,
    )
    tx_time_col = first_existing(tx, ["timestamp", "time", "datetime", "date", "step"])
    tx_amount_col = first_existing(tx, ["amount", "amt", "payment_amount", "amount_paid", "value"])
    tx_label_col = first_existing(tx, ["is_laundering", "label", "is_fraud", "fraud", "class"])

    persons = person.copy()
    persons["person_id"], person_map = _build_id_map(persons[person_id_col], offset=ACCOUNT_ID_OFFSET)
    if person_id_col != "person_id":
        persons = persons.drop(columns=[person_id_col])
    persons.insert(0, "person_id", persons.pop("person_id"))

    merchants = merchant.copy()
    if merchant_id_col is None:
        merchants["merchant_id"] = pd.RangeIndex(start=MERCHANT_ID_OFFSET, stop=MERCHANT_ID_OFFSET + len(merchants))
        merchant_map = {_normalize_key(i): MERCHANT_ID_OFFSET + i for i in range(len(merchants))}
    else:
        merchants["merchant_id"], merchant_map = _build_id_map(merchants[merchant_id_col], offset=MERCHANT_ID_OFFSET)
        if merchant_id_col != "merchant_id":
            merchants = merchants.drop(columns=[merchant_id_col])
    merchants.insert(0, "merchant_id", merchants.pop("merchant_id"))
    if "entity_type" not in merchants.columns:
        merchants["entity_type"] = "merchant"

    accounts = _build_accounts(persons)

    transactions = tx.copy()
    sender_raw = transactions[tx_sender_col].map(_normalize_key)
    transactions["sender_id"] = sender_raw.map(person_map)
    if transactions["sender_id"].isna().any():
        missing = sorted(sender_raw[transactions["sender_id"].isna()].unique().tolist())[:10]
        raise ValueError(f"Unmapped sender ids found in tx.csv: {missing}")

    if tx_receiver_col is None:
        transactions["receiver_id"] = -1
    else:
        receiver_raw = transactions[tx_receiver_col].map(_normalize_key)
        receiver_person = receiver_raw.map(person_map)
        receiver_merchant = receiver_raw.map(merchant_map)
        transactions["receiver_id"] = receiver_person.fillna(receiver_merchant).fillna(-1).astype("int64")

    if tx_id_col is None:
        transactions.insert(0, "transaction_id", range(len(transactions)))
    else:
        transactions["transaction_id"] = pd.Series(range(len(transactions)), index=transactions.index)

    time_values = transactions[tx_time_col]
    parsed_time = pd.to_datetime(time_values, utc=True, errors="coerce")
    if parsed_time.notna().sum() == 0 and pd.api.types.is_numeric_dtype(time_values):
        parsed_time = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(time_values.astype(float), unit="h")
    transactions["timestamp"] = parsed_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    transactions["amount"] = pd.to_numeric(transactions[tx_amount_col], errors="coerce")
    transactions["is_laundering"] = pd.to_numeric(transactions[tx_label_col], errors="coerce").fillna(0).astype("int64")

    for src_col, out_col, default in [
        (_pick_optional_column(tx, ["currency"]), "currency", "UNK"),
        (_pick_optional_column(tx, ["payment_format", "type", "channel"]), "payment_format", "UNK"),
        (_pick_optional_column(tx, ["sender_bank", "source_bank", "bankorig"]), "sender_bank", "UNK"),
        (_pick_optional_column(tx, ["receiver_bank", "dest_bank", "bankdest"]), "receiver_bank", "UNK"),
    ]:
        if src_col is None:
            transactions[out_col] = default
        else:
            transactions[out_col] = transactions[src_col].fillna(default).astype(str)

    keep_cols = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "currency",
        "payment_format",
        "sender_bank",
        "receiver_bank",
        "is_laundering",
    ]
    transactions = transactions[keep_cols].dropna(subset=["timestamp", "amount"]).copy()
    transactions["transaction_id"] = transactions["transaction_id"].astype("int64")
    transactions["sender_id"] = transactions["sender_id"].astype("int64")
    transactions["receiver_id"] = transactions["receiver_id"].astype("int64")

    accounts.to_csv(out_path / "accounts.csv", index=False)
    persons.to_csv(out_path / "persons.csv", index=False)
    merchants.to_csv(out_path / "merchants.csv", index=False)
    transactions.to_csv(out_path / "transactions.csv", index=False)

    return PreparedTransxionPaths(raw_dir=raw_path, out_dir=out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/transxion_public")
    parser.add_argument("--out_dir", default="data/raw/transxion_public_canonical")
    args = parser.parse_args()
    prepare_transxion_public_raw(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
