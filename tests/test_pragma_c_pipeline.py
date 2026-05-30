from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.assign_pragma_c_splits import assign_splits
from scripts.audit_pragma_c_graph import audit_graph
from scripts.audit_pragma_c_leakage import audit_leakage
from scripts.build_pragma_c_canonical_transactions import build_canonical_transactions
from scripts.build_pragma_c_eval_points import build_eval_points
from scripts.build_pragma_c_manifest import build_manifest
from scripts.build_pragma_c_tokenizer import build_tokenizer
from scripts.encode_pragma_c_records import encode_shard


def _write_raw_ibm_csv(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "LI-Medium_Trans.csv"
    df = pd.DataFrame(
        [
            {
                "Timestamp": "2024-01-01T00:00:00Z",
                "From Bank": "bank_a",
                "From Account": "acc_1",
                "To Bank": "bank_b",
                "To Account": "acc_2",
                "Amount Paid": 100.0,
                "Amount Received": 99.0,
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": "wire",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2024-01-02T00:00:00Z",
                "From Bank": "bank_a",
                "From Account": "acc_1",
                "To Bank": "bank_c",
                "To Account": "acc_3",
                "Amount Paid": 120.0,
                "Amount Received": 118.0,
                "Payment Currency": "USD",
                "Receiving Currency": "EUR",
                "Payment Format": "ach",
                "Is Laundering": 1,
            },
            {
                "Timestamp": "2024-01-03T00:00:00Z",
                "From Bank": "bank_b",
                "From Account": "acc_2",
                "To Bank": "bank_a",
                "To Account": "acc_1",
                "Amount Paid": 50.0,
                "Amount Received": 50.0,
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": "wire",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2024-01-04T00:00:00Z",
                "From Bank": "bank_c",
                "From Account": "acc_3",
                "To Bank": "bank_a",
                "To Account": "acc_1",
                "Amount Paid": 80.0,
                "Amount Received": 79.0,
                "Payment Currency": "EUR",
                "Receiving Currency": "USD",
                "Payment Format": "card",
                "Is Laundering": 1,
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def test_pragma_c_minimal_pipeline_builds_isolated_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "ibm_aml_li_medium_pragma_c"
    _write_raw_ibm_csv(raw_dir)

    canonical_path = build_canonical_transactions(raw_dir, output_root, raw_csv="LI-Medium_Trans.csv")
    eval_points_path = build_eval_points(output_root)
    assigned_eval_points_path = assign_splits(output_root)
    tokenizer_dir = build_tokenizer(output_root, profile_sample_limit=32, max_history_events=16)
    shard_dir = encode_shard(
        output_root,
        shard_index=0,
        num_shards=1,
        max_events=8,
        max_event_tokens=12,
        max_profile_tokens=16,
        max_history_events=16,
        max_eval_points_per_account_train=4,
        max_eval_points_per_account_valid=2,
        max_eval_points_per_account_calibration=2,
        lmdb_map_size_gb=1,
        lmdb_commit_interval=1,
    )
    manifest_path = build_manifest(output_root)
    leakage_path = audit_leakage(output_root)
    graph_path = audit_graph(output_root)

    assert canonical_path.exists()
    assert eval_points_path.exists()
    assert assigned_eval_points_path.exists()
    assert tokenizer_dir.joinpath("tokenizer.json").exists()
    assert shard_dir.joinpath("dataset.lmdb", "length.txt").exists()
    assert shard_dir.joinpath("train.lmdb", "length.txt").exists()
    assert shard_dir.joinpath("valid.lmdb", "length.txt").exists()
    assert manifest_path.exists()
    assert leakage_path.exists()
    assert graph_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_stage"] == "C"
    assert manifest["tokenizer_policy"] == "train_only_fit"
    assert manifest["shards"][0]["name"] == "shard_00000"

    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    assert leakage["passes"] is True

    shard_summary = json.loads((shard_dir / "shard_summary.json").read_text(encoding="utf-8"))
    assert shard_summary["split_mode"] == "pragma_c"
    assert shard_summary["max_events"] == 8
