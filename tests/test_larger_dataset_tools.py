from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.common.yaml_utils import load_yaml
from src.data_downloader.build_events import build_transxion_events
from src.tokenizer.build_vocab import main as build_vocab_main
from src.tokenizer.encode_dataset import main as encode_dataset_main
from tools.convert_elliptic_to_pralite import convert_elliptic_to_pralite
from tools.convert_paysim_to_pralite import convert_paysim_to_pralite
from tools.make_entity_event_cut import make_entity_event_cut
from tools.make_entity_splits import make_entity_splits
from tools.prepare_transxion_public_raw import prepare_transxion_public_raw


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
        "scripts/download_transxion_public.sh",
        "scripts/download_paysim.sh",
        "scripts/download_elliptic.sh",
        "scripts/download_hf_public_datasets.sh",
        "scripts/download_fdb.sh",
        "scripts/run_transxion_benchmark.sh",
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
            {"id": "p1", "age_bucket": "18_24", "region": "UK", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "p2", "age_bucket": "25_34", "region": "FR", "created_at": "2024-01-02T00:00:00Z"},
            {"id": "p3", "age_bucket": "35_44", "region": "DE", "created_at": "2024-01-03T00:00:00Z"},
        ]
    ).to_csv(raw_dir / "person.csv", index=False)
    pd.DataFrame(
        [
            {"id": "m1", "merchant_category": "grocery", "merchant_region": "UK"},
            {"id": "m2", "merchant_category": "travel", "merchant_region": "FR"},
        ]
    ).to_csv(raw_dir / "merchant.csv", index=False)
    pd.DataFrame(
        [
            {
                "tx_id": "t1",
                "from_id": "p1",
                "to_id": "m1",
                "timestamp": "2024-02-01T00:00:00Z",
                "amount": 10.0,
                "currency": "GBP",
                "type": "card",
                "sender_bank": "bank_a",
                "receiver_bank": "bank_b",
                "label": 0,
            },
            {
                "tx_id": "t2",
                "from_id": "p1",
                "to_id": "p2",
                "timestamp": "2024-02-01T01:00:00Z",
                "amount": 20.0,
                "currency": "GBP",
                "type": "bank_transfer",
                "sender_bank": "bank_a",
                "receiver_bank": "bank_c",
                "label": 1,
            },
            {
                "tx_id": "t3",
                "from_id": "p2",
                "to_id": "m2",
                "timestamp": "2024-02-02T02:00:00Z",
                "amount": 30.0,
                "currency": "EUR",
                "type": "card",
                "sender_bank": "bank_b",
                "receiver_bank": "bank_d",
                "label": 0,
            },
            {
                "tx_id": "t4",
                "from_id": "p3",
                "to_id": "m1",
                "timestamp": "2024-02-03T03:00:00Z",
                "amount": 40.0,
                "currency": "EUR",
                "type": "card",
                "sender_bank": "bank_d",
                "receiver_bank": "bank_b",
                "label": 0,
            },
        ]
    ).to_csv(raw_dir / "tx.csv", index=False)

    canonical_dir = tmp_path / "raw" / "transxion_public_canonical"
    prepare_transxion_public_raw(raw_dir, canonical_dir)

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
