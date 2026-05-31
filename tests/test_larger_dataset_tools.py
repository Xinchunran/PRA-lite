from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

from src.common.yaml_utils import load_yaml
from src.data_downloader.build_events import build_transxion_events
from src.tokenizer.build_vocab import main as build_vocab_main
from src.tokenizer.encode_dataset import main as encode_dataset_main
from src.training.data import load_tokenized_manifest_split
from tools.convert_elliptic_to_pralite import convert_elliptic_to_pralite
from tools.convert_ibm_aml_to_pralite import convert_ibm_aml_to_pralite
from tools.convert_paysim_to_pralite import convert_paysim_to_pralite
from tools.make_entity_event_cut import make_entity_event_cut
from tools.make_entity_splits import make_entity_splits
from tools.prepare_transxion_public_raw import prepare_transxion_public_raw
from tools.split_ibm_aml_csv import split_ibm_aml_csv


REQUIRED_TOKENIZED_COLUMNS = {
    "profile_key_ids",
    "profile_value_ids",
    "profile_value_pos",
    "profile_time",
    "profile_mask",
    "event_key_ids",
    "event_value_ids",
    "event_value_pos",
    "event_token_mask",
    "event_time",
    "calendar_features",
    "event_mask",
    "entity_id",
    "label",
    "evaluation_time",
}


def _tokenize_processed_dir(processed_dir: Path, monkeypatch, max_events: int = 8) -> pd.DataFrame:
    tokenizer_dir = processed_dir / "tokenizer"
    tokenized_dir = processed_dir / "tokenized"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vocab",
            "--processed_dir",
            str(processed_dir),
            "--output_dir",
            str(tokenizer_dir),
            "--num_buckets",
            "8",
            "--min_freq",
            "1",
        ],
    )
    build_vocab_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encode_dataset",
            "--processed_dir",
            str(processed_dir),
            "--tokenizer_dir",
            str(tokenizer_dir),
            "--output_dir",
            str(tokenized_dir),
            "--max_events",
            str(max_events),
            "--max_event_tokens",
            "12",
            "--max_profile_tokens",
            "32",
        ],
    )
    encode_dataset_main()
    return pd.read_parquet(tokenized_dir / "dataset.parquet")


def _assert_split_outputs(labels_path: Path, split_dir: Path) -> None:
    labels = pd.read_parquet(labels_path)
    all_ids = set(labels["entity_id"].tolist())

    seen: set[int] = set()
    for name in ["train", "valid", "test"]:
        csv_path = split_dir / f"{name}.csv"
        txt_path = split_dir / f"{name}_ids.txt"
        assert csv_path.exists()
        assert txt_path.exists()

        csv_ids = set(pd.read_csv(csv_path)["entity_id"].tolist()) if csv_path.stat().st_size > 0 else set()
        txt_ids = {
            int(line.strip())
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert csv_ids == txt_ids
        assert seen.isdisjoint(csv_ids)
        seen |= csv_ids

    assert seen == all_ids


def test_larger_dataset_scripts_and_config_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "scripts/download/download_transxion_public.sh",
        "scripts/download/download_paysim.sh",
        "scripts/download/download_elliptic.sh",
        "scripts/download/download_hf_public_datasets.sh",
        "scripts/download/download_fdb.sh",
        "scripts/benchmarks/run_transxion_benchmark.sh",
        "configs/data/transxion_public.yaml",
        "configs/data/transxion_full.yaml",
        "configs/train/pretrain_mlm_mini.yaml",
        "configs/train/pretrain_mlm_small.yaml",
    ]:
        assert (root / relative).exists(), relative

    cfg = load_yaml(root / "configs" / "data" / "transxion_public.yaml")
    assert cfg["dataset"] == "transxion"
    assert cfg["label_col"] not in cfg["transaction_columns"]


def test_transxion_public_adapter_builds_processed_cut_split_and_tokenized(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw" / "transxion_public"
    raw_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "person_id": "p1",
                "bank_account_number": "A000001",
                "bank": "1001",
                "person_age": 23,
                "person_education": "college",
                "person_gender": "female",
                "person_marital_status": "single",
                "person_occupation": "student",
            },
            {
                "person_id": "p2",
                "bank_account_number": "A000002",
                "bank": "1002",
                "person_age": 35,
                "person_education": "bachelor",
                "person_gender": "male",
                "person_marital_status": "married",
                "person_occupation": "engineer",
            },
            {
                "person_id": "p3",
                "bank_account_number": "A000003",
                "bank": "1003",
                "person_age": 49,
                "person_education": "graduate",
                "person_gender": "female",
                "person_marital_status": "married",
                "person_occupation": "manager",
            },
        ]
    ).to_csv(raw_dir / "person.csv", index=False)
    pd.DataFrame(
        [
            {
                "merchant_id": "m1",
                "bank_account_number": "A010001",
                "bank": "2001",
                "description": "Retail merchant",
                "type": "Small",
                "registered_capital": 100000.0,
                "industry": "Retail",
                "operating_status": "Active",
                "establishment_date": "2010-01-01",
                "legal_representative_id": "p1",
            },
            {
                "merchant_id": "m2",
                "bank_account_number": "A010002",
                "bank": "2002",
                "description": "Travel merchant",
                "type": "Medium",
                "registered_capital": 250000.0,
                "industry": "Travel",
                "operating_status": "Active",
                "establishment_date": "2011-06-01",
                "legal_representative_id": "p2",
            },
        ]
    ).to_csv(raw_dir / "merchant.csv", index=False)
    pd.DataFrame(
        [
            {
                "Timestamp": "2024-02-01 00:00:00",
                "From Bank": "1001",
                "From Account": "A000001",
                "To Bank": "2001",
                "To Account": "A010001",
                "Amount Received": 9.8,
                "Receiving Currency": "GBP",
                "Amount Paid": 10.0,
                "Payment Currency": "GBP",
                "Payment Format": "card",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2024-02-01 01:00:00",
                "From Bank": "1001",
                "From Account": "A000001",
                "To Bank": "1002",
                "To Account": "A000002",
                "Amount Received": 19.8,
                "Receiving Currency": "GBP",
                "Amount Paid": 20.0,
                "Payment Currency": "GBP",
                "Payment Format": "bank_transfer",
                "Is Laundering": 1,
            },
            {
                "Timestamp": "2024-02-02 02:00:00",
                "From Bank": "1002",
                "From Account": "A000002",
                "To Bank": "2002",
                "To Account": "A010002",
                "Amount Received": 29.0,
                "Receiving Currency": "EUR",
                "Amount Paid": 30.0,
                "Payment Currency": "EUR",
                "Payment Format": "card",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2024-02-03 03:00:00",
                "From Bank": "1003",
                "From Account": "A000003",
                "To Bank": "2001",
                "To Account": "A010001",
                "Amount Received": 39.5,
                "Receiving Currency": "EUR",
                "Amount Paid": 40.0,
                "Payment Currency": "EUR",
                "Payment Format": "card",
                "Is Laundering": 0,
            },
        ]
    ).to_csv(raw_dir / "tx.csv", index=False)

    canonical_dir = tmp_path / "raw" / "transxion_public_canonical"
    prepare_transxion_public_raw(raw_dir, canonical_dir)
    accounts_raw = pd.read_csv(canonical_dir / "accounts.csv")
    transactions_raw = pd.read_csv(canonical_dir / "transactions.csv")
    assert {"person", "merchant"}.issubset(set(accounts_raw["entity_type"]))
    assert {"sender_id", "receiver_id", "sender_bank", "receiver_bank"}.issubset(transactions_raw.columns)

    config_path = tmp_path / "transxion_public.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: transxion",
                f"raw_dir: {canonical_dir}",
                f"processed_dir: {tmp_path / 'processed' / 'transxion_public'}",
                "entity_id_col: account_id",
                "timestamp_col: timestamp",
                "label_col: is_laundering",
                "profile_files:",
                "  - accounts.csv",
                "  - persons.csv",
                "  - merchants.csv",
                "transaction_file: transactions.csv",
                "transaction_columns:",
                "  - timestamp",
                "  - sender_id",
                "  - receiver_id",
                "  - amount",
                "  - currency",
                "  - payment_format",
                "  - sender_bank",
                "  - receiver_bank",
                "",
            ]
        ),
        encoding="utf-8",
    )
    build_transxion_events(config_path)

    processed_dir = tmp_path / "processed" / "transxion_public"
    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    events = pd.read_parquet(processed_dir / "events.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")

    assert {"entity_id"}.issubset(profiles.columns)
    assert {"entity_id", "event_id", "timestamp"}.issubset(events.columns)
    assert {"entity_id", "label", "evaluation_time"}.issubset(labels.columns)
    assert "is_laundering" not in events.columns

    cut_dir = tmp_path / "processed" / "transxion_200k"
    make_entity_event_cut(processed_dir, cut_dir, target_events=3, seed=7)
    cut_events = pd.read_parquet(cut_dir / "events.parquet")
    assert len(cut_events) <= 3

    split_dir = tmp_path / "splits" / "transxion_200k"
    make_entity_splits(cut_dir / "labels.parquet", split_dir, seed=7, train_frac=0.5, valid_frac=0.25)
    _assert_split_outputs(cut_dir / "labels.parquet", split_dir)

    tokenized = _tokenize_processed_dir(cut_dir, monkeypatch, max_events=4)
    assert REQUIRED_TOKENIZED_COLUMNS.issubset(tokenized.columns)
    assert len(tokenized) > 0


def test_convert_paysim_pipeline_preserves_schema_and_excludes_label_leakage(tmp_path: Path, monkeypatch) -> None:
    raw_csv = tmp_path / "paysim.csv"
    pd.DataFrame(
        [
            {
                "step": 1,
                "type": "CASH_IN",
                "amount": 15.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 115.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 15.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 2,
                "type": "TRANSFER",
                "amount": 40.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 115.0,
                "newbalanceOrig": 75.0,
                "nameDest": "C2",
                "oldbalanceDest": 30.0,
                "newbalanceDest": 70.0,
                "isFraud": 1,
                "isFlaggedFraud": 1,
            },
            {
                "step": 3,
                "type": "PAYMENT",
                "amount": 20.0,
                "nameOrig": "C2",
                "oldbalanceOrg": 80.0,
                "newbalanceOrig": 60.0,
                "nameDest": "M2",
                "oldbalanceDest": 10.0,
                "newbalanceDest": 30.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 4,
                "type": "TRANSFER",
                "amount": 12.0,
                "nameOrig": "C3",
                "oldbalanceOrg": 50.0,
                "newbalanceOrig": 38.0,
                "nameDest": "C1",
                "oldbalanceDest": 75.0,
                "newbalanceDest": 87.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 5,
                "type": "PAYMENT",
                "amount": 8.0,
                "nameOrig": "C3",
                "oldbalanceOrg": 38.0,
                "newbalanceOrig": 30.0,
                "nameDest": "M1",
                "oldbalanceDest": 15.0,
                "newbalanceDest": 23.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            },
        ]
    ).to_csv(raw_csv, index=False)

    processed_dir = tmp_path / "processed" / "paysim_full"
    convert_paysim_to_pralite(raw_csv, processed_dir)
    events = pd.read_parquet(processed_dir / "events.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")

    assert {"entity_id", "event_id", "timestamp"}.issubset(events.columns)
    assert "isFraud" not in events.columns
    assert "isFlaggedFraud" not in events.columns
    assert {"entity_id", "label", "evaluation_time"}.issubset(labels.columns)
    assert labels["label"].sum() == 1

    cut_dir = tmp_path / "processed" / "paysim_2m"
    make_entity_event_cut(processed_dir, cut_dir, target_events=4, seed=3)
    assert len(pd.read_parquet(cut_dir / "events.parquet")) <= 4

    split_dir = tmp_path / "splits" / "paysim_2m"
    make_entity_splits(cut_dir / "labels.parquet", split_dir, seed=3, train_frac=0.5, valid_frac=0.25)
    _assert_split_outputs(cut_dir / "labels.parquet", split_dir)

    tokenized = _tokenize_processed_dir(cut_dir, monkeypatch, max_events=4)
    assert REQUIRED_TOKENIZED_COLUMNS.issubset(tokenized.columns)
    assert len(tokenized) > 0


def test_convert_elliptic_pipeline_generates_labeled_graph_adaptation(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw" / "elliptic"
    raw_dir.mkdir(parents=True)

    feature_rows = []
    for tx_id, timestep, base in [("tx1", 1, 0.1), ("tx2", 2, 0.2), ("tx3", 3, 0.3), ("tx4", 4, 0.4)]:
        feature_rows.append([tx_id, timestep] + [base + (i * 0.01) for i in range(165)])
    pd.DataFrame(feature_rows).to_csv(raw_dir / "elliptic_txs_features.csv", index=False, header=False)
    pd.DataFrame(
        [
            {"txId": "tx1", "class": "1"},
            {"txId": "tx2", "class": "2"},
            {"txId": "tx3", "class": "unknown"},
            {"txId": "tx4", "class": "1"},
        ]
    ).to_csv(raw_dir / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame(
        [
            {"txId1": "tx1", "txId2": "tx2"},
            {"txId1": "tx2", "txId2": "tx4"},
            {"txId1": "tx3", "txId2": "tx1"},
        ]
    ).to_csv(raw_dir / "elliptic_txs_edgelist.csv", index=False)

    processed_dir = tmp_path / "processed" / "elliptic_200k"
    convert_elliptic_to_pralite(raw_dir, processed_dir)

    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    events = pd.read_parquet(processed_dir / "events.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")

    assert len(labels) == 3
    assert {"entity_id", "event_id", "timestamp"}.issubset(events.columns)
    assert {"entity_id", "label", "evaluation_time"}.issubset(labels.columns)
    assert set(events["entity_id"]).issubset(set(labels["entity_id"]))
    assert set(profiles["entity_id"]) == set(labels["entity_id"])

    split_dir = tmp_path / "splits" / "elliptic_200k"
    make_entity_splits(processed_dir / "labels.parquet", split_dir, seed=11, train_frac=0.5, valid_frac=0.25)
    _assert_split_outputs(processed_dir / "labels.parquet", split_dir)

    tokenized = _tokenize_processed_dir(processed_dir, monkeypatch, max_events=6)
    assert REQUIRED_TOKENIZED_COLUMNS.issubset(tokenized.columns)
    assert len(tokenized) == len(labels)


def test_convert_ibm_aml_pipeline_generates_account_histories(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw" / "ibm_aml"
    raw_dir.mkdir(parents=True)
    raw_csv = raw_dir / "LI-Small_Trans.csv"
    pd.DataFrame(
        [
            {
                "Timestamp": "2022-09-01 00:00:00",
                "From Bank": "B1",
                "From Account": "A1",
                "To Bank": "B2",
                "To Account": "A2",
                "Amount Paid": 10.0,
                "Amount Received": 10.0,
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": "WIRE",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2022-09-01 01:00:00",
                "From Bank": "B2",
                "From Account": "A2",
                "To Bank": "B3",
                "To Account": "A3",
                "Amount Paid": 12.5,
                "Amount Received": 12.0,
                "Payment Currency": "USD",
                "Receiving Currency": "EUR",
                "Payment Format": "ACH",
                "Is Laundering": 1,
            },
            {
                "Timestamp": "2022-09-01 02:00:00",
                "From Bank": "B1",
                "From Account": "A1",
                "To Bank": "B3",
                "To Account": "A3",
                "Amount Paid": 5.0,
                "Amount Received": 5.0,
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": "CARD",
                "Is Laundering": 0,
            },
        ]
    ).to_csv(raw_csv, index=False)

    processed_dir = tmp_path / "processed" / "ibm_aml_full"
    convert_ibm_aml_to_pralite(raw_dir, processed_dir)

    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    events = pd.read_parquet(processed_dir / "events.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")

    assert {"entity_id", "event_id", "timestamp", "direction", "counterparty_id"}.issubset(events.columns)
    assert {"entity_id", "label", "evaluation_time"}.issubset(labels.columns)
    assert {"entity_id", "tx_count", "sent_tx_count", "recv_tx_count"}.issubset(profiles.columns)
    assert len(events) == 6
    assert labels["label"].sum() >= 1

    split_dir = tmp_path / "splits" / "ibm_aml_full"
    make_entity_splits(processed_dir / "labels.parquet", split_dir, seed=7, train_frac=0.5, valid_frac=0.25)
    _assert_split_outputs(processed_dir / "labels.parquet", split_dir)

    tokenized = _tokenize_processed_dir(processed_dir, monkeypatch, max_events=6)
    assert REQUIRED_TOKENIZED_COLUMNS.issubset(tokenized.columns)
    assert len(tokenized) == len(labels)


def test_ibm_aml_shards_and_manifest_loader_work(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw" / "ibm_aml"
    raw_dir.mkdir(parents=True)
    raw_csv = raw_dir / "LI-Medium_Trans.csv"
    rows = []
    for idx in range(24):
        rows.append(
            {
                "Timestamp": f"2022-09-01 {idx % 24:02d}:00:00",
                "From Bank": f"B{idx % 3}",
                "From Account": f"A{idx}",
                "To Bank": f"B{(idx + 1) % 3}",
                "To Account": f"A{idx + 100}",
                "Amount Paid": float(10 + idx),
                "Amount Received": float(9 + idx),
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": "WIRE" if idx % 2 == 0 else "ACH",
                "Is Laundering": int(idx % 5 == 0),
            }
        )
    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    shard_dir = tmp_path / "raw_shards"
    split_ibm_aml_csv(raw_csv, shard_dir, rows_per_shard=8)
    shard_paths = sorted(shard_dir.glob("shard_*.csv"))
    assert len(shard_paths) == 3

    tokenizer_dir = tmp_path / "tokenizer"
    tokenized_dirs: list[Path] = []
    for shard_idx, shard_path in enumerate(shard_paths):
        processed_dir = tmp_path / "processed" / shard_path.stem
        convert_ibm_aml_to_pralite(raw_dir, processed_dir, raw_csv=str(shard_path))
        if shard_idx == 0:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "build_vocab",
                    "--processed_dir",
                    str(processed_dir),
                    "--output_dir",
                    str(tokenizer_dir),
                    "--num_buckets",
                    "8",
                    "--min_freq",
                    "1",
                ],
            )
            build_vocab_main()
        tokenized_dir = tmp_path / "tokenized" / shard_path.stem
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "encode_dataset",
                "--processed_dir",
                str(processed_dir),
                "--tokenizer_dir",
                str(tokenizer_dir),
                "--output_dir",
                str(tokenized_dir),
                "--backend",
                "lmdb",
                "--max_events",
                "8",
                "--max_event_tokens",
                "12",
                "--max_profile_tokens",
                "32",
                "--hash_split_seed",
                "26",
                "--train_frac",
                "0.7",
                "--valid_frac",
                "0.2",
            ],
        )
        encode_dataset_main()
        tokenized_dirs.append(tokenized_dir)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tokenizer_dir": str(tokenizer_dir),
                "shards": [
                    {"name": path.name, "tokenized_dir": str(path), "status": "ready"} for path in tokenized_dirs
                ],
            }
        ),
        encoding="utf-8",
    )

    train_ds = load_tokenized_manifest_split(manifest_path, "train")
    valid_ds = load_tokenized_manifest_split(manifest_path, "valid")
    assert len(train_ds) > 0
    assert len(valid_ds) > 0
