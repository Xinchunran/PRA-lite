from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.fs import ensure_dir


DEFAULT_CREATED_AT = "1970-01-01T00:00:00Z"


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


def _build_account_join_key(bank: object, account: object) -> str:
    return f"{_normalize_key(bank)}::{_normalize_key(account)}"


def _build_id_map(values: pd.Series, offset: int = 0) -> tuple[pd.Series, dict[str, int]]:
    normalized = values.map(_normalize_key)
    uniques = pd.Index(pd.unique(normalized))
    mapping = {key: offset + idx for idx, key in enumerate(uniques)}
    numeric = normalized.map(mapping).astype("int64")
    return numeric, mapping


def _pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return first_existing(df, candidates, required=False)


def _format_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna(DEFAULT_CREATED_AT)


def _age_to_bucket(values: pd.Series) -> pd.Series:
    age = pd.to_numeric(values, errors="coerce")
    buckets = pd.Series("unknown", index=values.index, dtype="string")
    ranges = [
        ((18, 24), "18_24"),
        ((25, 34), "25_34"),
        ((35, 44), "35_44"),
        ((45, 54), "45_54"),
        ((55, 64), "55_64"),
        ((65, 200), "65_plus"),
    ]
    for (lo, hi), label in ranges:
        buckets.loc[age.between(lo, hi, inclusive="both")] = label
    return buckets


def _prepare_id_based_schema(person: pd.DataFrame, merchant: pd.DataFrame, tx: pd.DataFrame, out_path: Path) -> None:
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
    persons["person_id"], person_map = _build_id_map(persons[person_id_col])
    if person_id_col != "person_id":
        persons = persons.drop(columns=[person_id_col])
    persons.insert(0, "person_id", persons.pop("person_id"))

    merchants = merchant.copy()
    if merchant_id_col is None:
        merchants["merchant_id"] = pd.RangeIndex(start=10_000_000, stop=10_000_000 + len(merchants))
        merchant_map = {_normalize_key(i): 10_000_000 + i for i in range(len(merchants))}
    else:
        merchants["merchant_id"], merchant_map = _build_id_map(merchants[merchant_id_col], offset=10_000_000)
        if merchant_id_col != "merchant_id":
            merchants = merchants.drop(columns=[merchant_id_col])
    merchants.insert(0, "merchant_id", merchants.pop("merchant_id"))
    if "entity_type" not in merchants.columns:
        merchants["entity_type"] = "merchant"

    region_col = _pick_optional_column(persons, ["region", "home_region", "country", "city"])
    created_col = _pick_optional_column(persons, ["created_at", "created_time", "signup_time", "register_time"])
    accounts = pd.DataFrame(
        {
            "account_id": persons["person_id"].astype("int64"),
            "person_id": persons["person_id"].astype("int64"),
            "entity_type": "person",
            "account_region": (
                persons[region_col].fillna("unknown").astype(str) if region_col else pd.Series("unknown", index=persons.index)
            ),
            "created_at": _format_timestamp(persons[created_col]) if created_col else pd.Series(DEFAULT_CREATED_AT, index=persons.index),
        }
    )

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
        transactions[out_col] = transactions[src_col].fillna(default).astype(str) if src_col else default

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


def _prepare_account_based_schema(person: pd.DataFrame, merchant: pd.DataFrame, tx: pd.DataFrame, out_path: Path) -> None:
    person_id_col = first_existing(person, ["person_id", "id"])
    person_bank_col = first_existing(person, ["bank", "bank_id"])
    person_account_col = first_existing(person, ["bank_account_number", "account_number", "account"])

    merchant_id_col = first_existing(merchant, ["merchant_id", "id"])
    merchant_bank_col = first_existing(merchant, ["bank", "bank_id"])
    merchant_account_col = first_existing(merchant, ["bank_account_number", "account_number", "account"])

    tx_time_col = first_existing(tx, ["Timestamp", "timestamp", "time"])
    tx_from_bank_col = first_existing(tx, ["From Bank", "sender_bank", "source_bank"])
    tx_from_account_col = first_existing(tx, ["From Account", "sender_account", "source_account"])
    tx_to_bank_col = first_existing(tx, ["To Bank", "receiver_bank", "dest_bank"])
    tx_to_account_col = first_existing(tx, ["To Account", "receiver_account", "dest_account"])
    tx_amount_paid_col = first_existing(tx, ["Amount Paid", "amount_paid", "amount"])
    tx_payment_currency_col = first_existing(tx, ["Payment Currency", "payment_currency", "currency"])
    tx_payment_format_col = first_existing(tx, ["Payment Format", "payment_format", "type"])
    tx_label_col = first_existing(tx, ["Is Laundering", "is_laundering", "label"])

    persons = person.copy()
    persons["source_person_id"] = persons[person_id_col].map(_normalize_key)
    persons["person_id"] = pd.RangeIndex(len(persons), dtype="int64")
    persons["account_key"] = [
        _build_account_join_key(bank, account)
        for bank, account in zip(persons[person_bank_col], persons[person_account_col], strict=False)
    ]
    age_col = _pick_optional_column(persons, ["person_age", "age", "customer_age"])
    persons["age_bucket"] = _age_to_bucket(persons[age_col]) if age_col else "unknown"
    persons["region"] = persons[person_bank_col].map(_normalize_key)
    persons = persons[["person_id", "source_person_id", "age_bucket", "region"]].copy()

    merchants = merchant.copy()
    merchants["source_merchant_id"] = merchants[merchant_id_col].map(_normalize_key)
    merchants["account_key"] = [
        _build_account_join_key(bank, account)
        for bank, account in zip(merchants[merchant_bank_col], merchants[merchant_account_col], strict=False)
    ]
    created_col = _pick_optional_column(merchants, ["establishment_date", "created_at"])
    merchants["created_at"] = _format_timestamp(merchants[created_col]) if created_col else DEFAULT_CREATED_AT

    tx_keys = pd.DataFrame(
        {
            "account_key": pd.concat(
                [
                    pd.Series(
                        [
                            _build_account_join_key(bank, account)
                            for bank, account in zip(tx[tx_from_bank_col], tx[tx_from_account_col], strict=False)
                        ]
                    ),
                    pd.Series(
                        [
                            _build_account_join_key(bank, account)
                            for bank, account in zip(tx[tx_to_bank_col], tx[tx_to_account_col], strict=False)
                        ]
                    ),
                ],
                ignore_index=True,
            )
        }
    ).drop_duplicates(ignore_index=True)
    tx_keys["account_id"] = pd.RangeIndex(len(tx_keys), dtype="int64")

    person_registry = pd.DataFrame(
        {
            "account_key": [
                _build_account_join_key(bank, account)
                for bank, account in zip(person[person_bank_col], person[person_account_col], strict=False)
            ],
            "person_id": persons["person_id"].astype("int64"),
            "entity_type": "person",
            "account_region": person[person_bank_col].map(_normalize_key),
            "created_at": DEFAULT_CREATED_AT,
        }
    )
    merchant_registry = pd.DataFrame(
        {
            "account_key": merchants["account_key"],
            "person_id": -(pd.RangeIndex(start=1, stop=len(merchants) + 1, dtype="int64")),
            "entity_type": "merchant",
            "account_region": merchant[merchant_bank_col].map(_normalize_key),
            "created_at": merchants["created_at"].astype(str),
        }
    )
    registry = tx_keys.merge(pd.concat([person_registry, merchant_registry], ignore_index=True), on="account_key", how="left")
    unknown_mask = registry["entity_type"].isna()
    if unknown_mask.any():
        n_unknown = int(unknown_mask.sum())
        registry.loc[unknown_mask, "entity_type"] = "unknown"
        registry.loc[unknown_mask, "account_region"] = registry.loc[unknown_mask, "account_key"].str.split("::").str[0]
        registry.loc[unknown_mask, "created_at"] = DEFAULT_CREATED_AT
        registry.loc[unknown_mask, "person_id"] = -(len(merchants) + pd.RangeIndex(start=1, stop=n_unknown + 1, dtype="int64"))
    registry["person_id"] = registry["person_id"].astype("int64")

    accounts = registry[["account_id", "person_id", "entity_type", "account_region", "created_at"]].copy()

    account_id_map = registry.set_index("account_key")["account_id"]
    transactions = tx.copy()
    transactions["sender_bank"] = transactions[tx_from_bank_col].map(_normalize_key)
    transactions["receiver_bank"] = transactions[tx_to_bank_col].map(_normalize_key)
    transactions["sender_id"] = [
        int(account_id_map[_build_account_join_key(bank, account)])
        for bank, account in zip(tx[tx_from_bank_col], tx[tx_from_account_col], strict=False)
    ]
    transactions["receiver_id"] = [
        int(account_id_map[_build_account_join_key(bank, account)])
        for bank, account in zip(tx[tx_to_bank_col], tx[tx_to_account_col], strict=False)
    ]
    transactions["timestamp"] = _format_timestamp(transactions[tx_time_col])
    transactions["amount"] = pd.to_numeric(transactions[tx_amount_paid_col], errors="coerce")
    transactions["currency"] = transactions[tx_payment_currency_col].fillna("UNK").astype(str)
    transactions["payment_format"] = transactions[tx_payment_format_col].fillna("UNK").astype(str)
    transactions["is_laundering"] = pd.to_numeric(transactions[tx_label_col], errors="coerce").fillna(0).astype("int64")
    transactions.insert(0, "transaction_id", range(len(transactions)))
    transactions = transactions[
        [
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
    ].dropna(subset=["timestamp", "amount"])

    merchants_out = merchants.drop(columns=["account_key"]).copy()
    if "merchant_id" in merchants_out.columns:
        merchants_out = merchants_out.rename(columns={"merchant_id": "source_merchant_id_raw"})
    merchants_out.insert(0, "merchant_id", pd.RangeIndex(start=10_000_000, stop=10_000_000 + len(merchants_out)))

    accounts.to_csv(out_path / "accounts.csv", index=False)
    persons.to_csv(out_path / "persons.csv", index=False)
    merchants_out.to_csv(out_path / "merchants.csv", index=False)
    transactions.to_csv(out_path / "transactions.csv", index=False)


def prepare_transxion_public_raw(raw_dir: str | Path, out_dir: str | Path) -> PreparedTransxionPaths:
    raw_path = Path(raw_dir)
    out_path = ensure_dir(out_dir)

    person = pd.read_csv(raw_path / "person.csv")
    merchant = pd.read_csv(raw_path / "merchant.csv")
    tx = pd.read_csv(raw_path / "tx.csv")

    uses_account_pairs = {
        "bank_account_number",
        "bank",
    }.issubset({str(col) for col in person.columns}) and {
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
    }.issubset({str(col) for col in tx.columns})

    if uses_account_pairs:
        _prepare_account_based_schema(person, merchant, tx, out_path)
    else:
        _prepare_id_based_schema(person, merchant, tx, out_path)

    return PreparedTransxionPaths(raw_dir=raw_path, out_dir=out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/transxion_public")
    parser.add_argument("--out_dir", default="data/raw/transxion_public_canonical")
    args = parser.parse_args()
    prepare_transxion_public_raw(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
