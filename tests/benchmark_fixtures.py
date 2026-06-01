from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel
from src.training.checkpoint import save_checkpoint


def write_tokenizer(tokenizer_dir: Path) -> None:
    payload = {
        "token_to_id": {
            "[PAD]": 0,
            "[UNK]": 1,
            "[MASK]": 2,
            "[USR]": 3,
            "[EVT]": 4,
            "K:P:region": 5,
            "V:P:region=uk": 6,
            "V:P:region=fr": 7,
            "K:E:merchant": 8,
            "V:E:merchant=a": 9,
            "V:E:merchant=b": 10,
            "V:E:merchant=c": 11,
        },
        "profile_cols": ["region"],
        "event_cols": ["merchant"],
        "numeric_binners": {},
    }
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_tokenized_dataset(data_dir: Path) -> None:
    rows = [
        {
            "entity_id": 1,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [6, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[9, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [2.0, 1.0, 0.0],
            "calendar_features": [[0.0] * 6, [1.0] * 6, [0.0] * 6],
            "event_mask": [1, 1, 0],
            "label": 0,
        },
        {
            "entity_id": 2,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [7, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[10, 0, 0, 0], [11, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [2.2, 0.8, 0.0],
            "calendar_features": [[1.0] * 6, [0.5] * 6, [0.0] * 6],
            "event_mask": [1, 1, 0],
            "label": 1,
        },
        {
            "entity_id": 3,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [6, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [1.5, 0.0, 0.0],
            "calendar_features": [[0.2] * 6, [0.0] * 6, [0.0] * 6],
            "event_mask": [1, 0, 0],
            "label": 0,
        },
        {
            "entity_id": 4,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [7, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [1.1, 0.0, 0.0],
            "calendar_features": [[0.6] * 6, [0.0] * 6, [0.0] * 6],
            "event_mask": [1, 0, 0],
            "label": 1,
        },
        {
            "entity_id": 5,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [6, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [1.7, 0.0, 0.0],
            "calendar_features": [[0.1] * 6, [0.0] * 6, [0.0] * 6],
            "event_mask": [1, 0, 0],
            "label": 0,
        },
        {
            "entity_id": 6,
            "profile_key_ids": [5, 0, 0, 0],
            "profile_value_ids": [7, 0, 0, 0],
            "profile_value_pos": [0, 0, 0, 0],
            "profile_time": [0.0, 0.0, 0.0, 0.0],
            "profile_mask": [1, 0, 0, 0],
            "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_ids": [[11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "event_time": [1.3, 0.0, 0.0],
            "calendar_features": [[0.8] * 6, [0.0] * 6, [0.0] * 6],
            "event_mask": [1, 0, 0],
            "label": 1,
        },
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / "dataset.parquet", index=False)


def write_splits(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_ids.txt").write_text("1\n2\n3\n4\n", encoding="utf-8")
    (split_dir / "valid_ids.txt").write_text("5\n", encoding="utf-8")
    (split_dir / "test_ids.txt").write_text("6\n", encoding="utf-8")


def write_checkpoint(ckpt_path: Path, tokenizer_dir: Path) -> None:
    cfg = PragmaLiteConfig(
        vocab_size=16,
        d_model=16,
        n_heads=4,
        d_ffn=32,
        n_layers=1,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
        max_profile_tokens=4,
        max_event_tokens=4,
        max_events=3,
        dropout=0.0,
    )
    model = PragmaLiteModel(cfg)
    save_checkpoint(
        ckpt_path,
        {
            "model_cfg": cfg.__dict__,
            "model_state": model.state_dict(),
            "tokenizer_dir": str(tokenizer_dir),
        },
    )


def write_processed_transxion_like_dir(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"entity_id": 1, "region": "uk", "segment": "consumer", "age_bucket": "18_24"},
            {"entity_id": 2, "region": "fr", "segment": "consumer", "age_bucket": "25_34"},
            {"entity_id": 3, "region": "uk", "segment": "merchant", "age_bucket": "35_44"},
            {"entity_id": 4, "region": "fr", "segment": "merchant", "age_bucket": "45_54"},
            {"entity_id": 5, "region": "uk", "segment": "consumer", "age_bucket": "18_24"},
            {"entity_id": 6, "region": "fr", "segment": "merchant", "age_bucket": "35_44"},
        ]
    ).to_parquet(processed_dir / "profiles.parquet", index=False)
    pd.DataFrame(
        [
            {"entity_id": 1, "event_id": 10, "timestamp": "2024-02-01T00:00:00Z", "amount": 10.0, "payment_format": "card", "currency": "GBP"},
            {"entity_id": 1, "event_id": 11, "timestamp": "2024-02-01T01:00:00Z", "amount": 20.0, "payment_format": "bank_transfer", "currency": "GBP"},
            {"entity_id": 2, "event_id": 12, "timestamp": "2024-02-02T00:00:00Z", "amount": 30.0, "payment_format": "card", "currency": "EUR"},
            {"entity_id": 3, "event_id": 13, "timestamp": "2024-02-03T00:00:00Z", "amount": 15.0, "payment_format": "cash", "currency": "GBP"},
            {"entity_id": 4, "event_id": 14, "timestamp": "2024-02-04T00:00:00Z", "amount": 35.0, "payment_format": "bank_transfer", "currency": "EUR"},
            {"entity_id": 5, "event_id": 15, "timestamp": "2024-02-05T00:00:00Z", "amount": 12.0, "payment_format": "card", "currency": "GBP"},
            {"entity_id": 6, "event_id": 16, "timestamp": "2024-02-06T00:00:00Z", "amount": 38.0, "payment_format": "bank_transfer", "currency": "EUR"},
        ]
    ).to_parquet(processed_dir / "events.parquet", index=False)
    pd.DataFrame(
        [
            {"entity_id": 1, "label": 0, "evaluation_time": "2024-02-10T00:00:00Z"},
            {"entity_id": 2, "label": 1, "evaluation_time": "2024-02-10T00:00:00Z"},
            {"entity_id": 3, "label": 0, "evaluation_time": "2024-02-10T00:00:00Z"},
            {"entity_id": 4, "label": 1, "evaluation_time": "2024-02-10T00:00:00Z"},
            {"entity_id": 5, "label": 0, "evaluation_time": "2024-02-10T00:00:00Z"},
            {"entity_id": 6, "label": 1, "evaluation_time": "2024-02-10T00:00:00Z"},
        ]
    ).to_parquet(processed_dir / "labels.parquet", index=False)
