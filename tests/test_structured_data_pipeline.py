from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from src.common.tokenized_lmdb import TokenizedLmdbWriter
from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.encode_dataset import main as encode_dataset_main
from src.training.checkpoint import load_checkpoint
from src.training.data import LmdbTokenizedDataset, TokenizedDataset, load_tokenized_dataset, load_tokenized_split, pad_collate
from src.training.pretrain_mlm import main as pretrain_main


def _write_tokenizer(tokenizer_dir: Path) -> None:
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
    (tokenizer_dir / "tokenizer.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_processed_tables(processed_dir: Path) -> None:
    profiles = pd.DataFrame(
        [
            {"entity_id": 1, "region": "uk"},
            {"entity_id": 2, "region": "fr"},
            {"entity_id": 3, "region": "uk"},
        ]
    )
    events = pd.DataFrame(
        [
            {"entity_id": 1, "event_id": 1, "timestamp": "2024-04-29T10:00:00Z", "merchant": "a"},
            {"entity_id": 1, "event_id": 2, "timestamp": "2024-04-30T10:00:00Z", "merchant": "b"},
            {"entity_id": 2, "event_id": 3, "timestamp": "2024-04-28T12:00:00Z", "merchant": "b"},
            {"entity_id": 2, "event_id": 4, "timestamp": "2024-04-30T13:00:00Z", "merchant": "c"},
            {"entity_id": 3, "event_id": 5, "timestamp": "2024-04-27T08:00:00Z", "merchant": "a"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"entity_id": 1, "label": 0, "evaluation_time": "2024-05-01T00:00:00Z"},
            {"entity_id": 2, "label": 1, "evaluation_time": "2024-05-01T00:00:00Z"},
            {"entity_id": 3, "label": 0, "evaluation_time": "2024-05-01T00:00:00Z"},
        ]
    )
    profiles.to_parquet(processed_dir / "profiles.parquet", index=False)
    events.to_parquet(processed_dir / "events.parquet", index=False)
    labels.to_parquet(processed_dir / "labels.parquet", index=False)


def _write_split_ids(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_ids.txt").write_text("1\n2\n", encoding="utf-8")
    (split_dir / "valid_ids.txt").write_text("3\n", encoding="utf-8")
    (split_dir / "test_ids.txt").write_text("2\n", encoding="utf-8")


def test_encode_dataset_writes_structured_columns(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "tokenized"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)

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
            str(output_dir),
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
        ],
    )
    encode_dataset_main()

    df = pd.read_parquet(output_dir / "dataset.parquet")
    assert {
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
    }.issubset(df.columns)
    assert len(df.loc[0, "profile_key_ids"]) == 4
    assert len(df.loc[0, "event_key_ids"]) == 3
    assert len(df.loc[0, "event_key_ids"][0]) == 4
    assert len(df.loc[0, "calendar_features"][0]) == 6


def test_encode_dataset_parallel_matches_single_worker(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir_single = tmp_path / "tokenized_single"
    output_dir_parallel = tmp_path / "tokenized_parallel"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)

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
            str(output_dir_single),
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
            "--num_workers",
            "1",
        ],
    )
    encode_dataset_main()

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
            str(output_dir_parallel),
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
            "--num_workers",
            "2",
        ],
    )
    encode_dataset_main()

    df_single = pd.read_parquet(output_dir_single / "dataset.parquet").sort_values("entity_id").reset_index(drop=True)
    df_parallel = pd.read_parquet(output_dir_parallel / "dataset.parquet").sort_values("entity_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(df_single, df_parallel)


def test_encode_dataset_writes_split_parquets(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "tokenized"
    split_dir = tmp_path / "splits"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)
    _write_split_ids(split_dir)

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
            str(output_dir),
            "--split_dir",
            str(split_dir),
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
            "--row_group_size",
            "1",
        ],
    )
    encode_dataset_main()

    train_df = pd.read_parquet(output_dir / "train.parquet").sort_values("entity_id").reset_index(drop=True)
    valid_df = pd.read_parquet(output_dir / "valid.parquet").sort_values("entity_id").reset_index(drop=True)
    test_df = pd.read_parquet(output_dir / "test.parquet").sort_values("entity_id").reset_index(drop=True)
    assert train_df["entity_id"].tolist() == [1, 2]
    assert valid_df["entity_id"].tolist() == [3]
    assert test_df["entity_id"].tolist() == [2]


def test_encode_dataset_writes_lmdb_backend(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "tokenized"
    split_dir = tmp_path / "splits"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)
    _write_split_ids(split_dir)

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
            str(output_dir),
            "--split_dir",
            str(split_dir),
            "--backend",
            "lmdb",
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
            "--lmdb_map_size_gb",
            "1",
            "--lmdb_commit_interval",
            "1",
        ],
    )
    encode_dataset_main()

    assert (output_dir / "dataset.lmdb" / "data.mdb").exists()
    assert (output_dir / "train.lmdb" / "data.mdb").exists()
    ds = LmdbTokenizedDataset(output_dir / "dataset.lmdb")
    train_ds = load_tokenized_split(output_dir, "train", split_dir=split_dir)
    assert len(ds) == 3
    assert len(train_ds) == 2
    assert ds[0]["entity_id"] == 1
    assert train_ds[1]["entity_id"] == 2


def test_dataloader_returns_structured_batch_and_direct_model_inputs(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "tokenized"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)

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
            str(output_dir),
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
        ],
    )
    encode_dataset_main()

    ds = TokenizedDataset(output_dir / "dataset.parquet")
    batch = pad_collate([ds[0], ds[1]], pad_id=0)
    model_inputs = batch.model_inputs()
    assert set(model_inputs.keys()) == {
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
    }

    model = PragmaLiteModel(
        PragmaLiteConfig(
            vocab_size=16,
            d_model=32,
            n_heads=4,
            d_ffn=64,
            n_layers=1,
            profile_layers=1,
            event_layers=1,
            history_layers=1,
            max_profile_tokens=4,
            max_event_tokens=4,
            max_events=3,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        out = model(**model_inputs)
        logits = model(**model_inputs, return_mlm_logits=True)
    assert out["record_embedding"].shape == (2, model.d_model)
    assert logits.shape[:3] == batch.event_value_ids.shape


def test_tokenized_dataset_lazy_filters_rows_without_loading_full_dataframe(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    df = pd.DataFrame(
        [
            {
                "entity_id": 1,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0]],
                "event_value_ids": [[9, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0]],
                "event_time": [1.0],
                "calendar_features": [[0.0] * 6],
                "event_mask": [1],
                "label": 0,
            },
            {
                "entity_id": 2,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [7, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0]],
                "event_value_ids": [[10, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0]],
                "event_time": [1.0],
                "calendar_features": [[1.0] * 6],
                "event_mask": [1],
                "label": 1,
            },
            {
                "entity_id": 3,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0]],
                "event_value_ids": [[11, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0]],
                "event_time": [1.0],
                "calendar_features": [[0.5] * 6],
                "event_mask": [1],
                "label": 0,
            },
        ]
    )
    df.to_parquet(tokenized_dir / "dataset.parquet", index=False, row_group_size=1)

    ds = TokenizedDataset(tokenized_dir / "dataset.parquet", entity_ids={1, 3})
    assert len(ds) == 2
    assert ds[0]["entity_id"] == 1
    assert ds[1]["entity_id"] == 3
    assert ds[-1]["label"] == 0


def test_load_tokenized_split_prefers_split_parquet(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    split_dir = tmp_path / "splits"
    tokenized_dir.mkdir()
    split_dir.mkdir()
    pd.DataFrame(
        [
            {
                "entity_id": 99,
                "profile_key_ids": [5, 0],
                "profile_value_ids": [6, 0],
                "profile_value_pos": [0, 0],
                "profile_time": [0.0, 0.0],
                "profile_mask": [1, 0],
                "event_key_ids": [[8, 0]],
                "event_value_ids": [[9, 0]],
                "event_value_pos": [[0, 0]],
                "event_token_mask": [[1, 0]],
                "event_time": [1.0],
                "calendar_features": [[0.0] * 6],
                "event_mask": [1],
                "label": 1,
            }
        ]
    ).to_parquet(tokenized_dir / "train.parquet", index=False, row_group_size=1)
    (split_dir / "train_ids.txt").write_text("1\n2\n", encoding="utf-8")

    ds = load_tokenized_split(tokenized_dir, "train", split_dir=split_dir)
    assert len(ds) == 1
    assert ds[0]["entity_id"] == 99


def test_load_tokenized_dataset_prefers_lmdb_backend(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    writer = TokenizedLmdbWriter(tokenized_dir / "dataset.lmdb", map_size_gb=1, commit_interval=1)
    writer.write(
        {
            "entity_id": 7,
            "profile_key_ids": [5, 0],
            "profile_value_ids": [6, 0],
            "profile_value_pos": [0, 0],
            "profile_time": [0.0, 0.0],
            "profile_mask": [1, 0],
            "event_key_ids": [[8, 0]],
            "event_value_ids": [[9, 0]],
            "event_value_pos": [[0, 0]],
            "event_token_mask": [[1, 0]],
            "event_time": [1.0],
            "calendar_features": [[0.0] * 6],
            "event_mask": [1],
            "label": 1,
        }
    )
    writer.close()

    ds = load_tokenized_dataset(tokenized_dir)
    assert isinstance(ds, LmdbTokenizedDataset)
    assert len(ds) == 1
    assert ds[0]["entity_id"] == 7


def test_pretrain_pipeline_runs_on_structured_only_dataset(tmp_path: Path, monkeypatch) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenizer_dir = tmp_path / "tokenizer"
    split_dir = tmp_path / "splits"
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "runs"
    tokenized_dir.mkdir()
    tokenizer_dir.mkdir()
    split_dir.mkdir()
    config_dir.mkdir()

    _write_tokenizer(tokenizer_dir)
    df = pd.DataFrame(
        [
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
                "event_time": [1.0, 0.5, 0.0],
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
                "event_time": [1.2, 0.3, 0.0],
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
                "event_time": [0.8, 0.0, 0.0],
                "calendar_features": [[0.2] * 6, [0.0] * 6, [0.0] * 6],
                "event_mask": [1, 0, 0],
                "label": 0,
            },
        ]
    )
    df.to_parquet(tokenized_dir / "dataset.parquet", index=False)
    (split_dir / "train_ids.txt").write_text("1\n2\n", encoding="utf-8")
    (split_dir / "valid_ids.txt").write_text("3\n", encoding="utf-8")

    train_cfg = config_dir / "pretrain.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "training:",
                "  batch_size: 2",
                "  max_steps: 1",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 0.0",
                "  seed: 7",
                "  log_every: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "masking:",
                "  token_mask_prob: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    model_cfg = config_dir / "model.yaml"
    model_cfg.write_text(
        "\n".join(
            [
                "model:",
                "  d_model: 32",
                "  n_heads: 4",
                "  n_layers: 1",
                "  d_ffn: 64",
                "  dropout: 0.0",
                "  max_profile_tokens: 4",
                "  max_event_tokens: 4",
                "  max_events: 3",
                "  profile_layers: 1",
                "  event_layers: 1",
                "  history_layers: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_mlm",
            "--config",
            str(train_cfg),
            "--model_config",
            str(model_cfg),
            "--data_dir",
            str(tokenized_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(output_dir),
            "--tokenizer_dir",
            str(tokenizer_dir),
            "--device",
            "cpu",
        ],
    )
    pretrain_main()

    ckpt = load_checkpoint(output_dir / "best.ckpt", map_location="cpu")
    model = PragmaLiteModel(PragmaLiteConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch = pad_collate([TokenizedDataset(tokenized_dir / "dataset.parquet")[0]], pad_id=0)
    with torch.no_grad():
        logits = model(**batch.model_inputs(), return_mlm_logits=True)
    assert logits.shape == (1, 3, 4, model.vocab_size)


def test_pretrain_pipeline_uses_split_parquets_when_available(tmp_path: Path, monkeypatch) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenizer_dir = tmp_path / "tokenizer"
    split_dir = tmp_path / "splits"
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "runs"
    tokenized_dir.mkdir()
    tokenizer_dir.mkdir()
    split_dir.mkdir()
    config_dir.mkdir()

    _write_tokenizer(tokenizer_dir)
    train_df = pd.DataFrame(
        [
            {
                "entity_id": 11,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_time": [1.0, 0.0, 0.0],
                "calendar_features": [[0.0] * 6, [0.0] * 6, [0.0] * 6],
                "event_mask": [1, 0, 0],
                "label": 0,
            },
            {
                "entity_id": 12,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [7, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_time": [1.2, 0.0, 0.0],
                "calendar_features": [[1.0] * 6, [0.0] * 6, [0.0] * 6],
                "event_mask": [1, 0, 0],
                "label": 1,
            },
        ]
    )
    valid_df = pd.DataFrame(
        [
            {
                "entity_id": 13,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_time": [0.8, 0.0, 0.0],
                "calendar_features": [[0.2] * 6, [0.0] * 6, [0.0] * 6],
                "event_mask": [1, 0, 0],
                "label": 0,
            },
        ]
    )
    train_df.to_parquet(tokenized_dir / "train.parquet", index=False, row_group_size=1)
    valid_df.to_parquet(tokenized_dir / "valid.parquet", index=False, row_group_size=1)

    train_cfg = config_dir / "pretrain.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "training:",
                "  batch_size: 2",
                "  max_steps: 1",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 0.0",
                "  seed: 7",
                "  log_every: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "masking:",
                "  token_mask_prob: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    model_cfg = config_dir / "model.yaml"
    model_cfg.write_text(
        "\n".join(
            [
                "model:",
                "  d_model: 32",
                "  n_heads: 4",
                "  n_layers: 1",
                "  d_ffn: 64",
                "  dropout: 0.0",
                "  max_profile_tokens: 4",
                "  max_event_tokens: 4",
                "  max_events: 3",
                "  profile_layers: 1",
                "  event_layers: 1",
                "  history_layers: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_mlm",
            "--config",
            str(train_cfg),
            "--model_config",
            str(model_cfg),
            "--data_dir",
            str(tokenized_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(output_dir),
            "--tokenizer_dir",
            str(tokenizer_dir),
            "--device",
            "cpu",
        ],
    )
    pretrain_main()

    assert (output_dir / "best.ckpt").exists()


def test_pretrain_pipeline_uses_split_lmdb_when_available(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    tokenized_dir = tmp_path / "tokenized"
    tokenizer_dir = tmp_path / "tokenizer"
    split_dir = tmp_path / "splits"
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "runs"
    processed_dir.mkdir()
    tokenizer_dir.mkdir()
    split_dir.mkdir()
    config_dir.mkdir()
    _write_processed_tables(processed_dir)
    _write_tokenizer(tokenizer_dir)
    _write_split_ids(split_dir)

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
            "--split_dir",
            str(split_dir),
            "--backend",
            "lmdb",
            "--max_events",
            "3",
            "--max_event_tokens",
            "4",
            "--max_profile_tokens",
            "4",
            "--lmdb_map_size_gb",
            "1",
            "--lmdb_commit_interval",
            "1",
        ],
    )
    encode_dataset_main()

    train_cfg = config_dir / "pretrain.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "training:",
                "  batch_size: 2",
                "  max_steps: 1",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 0.0",
                "  seed: 7",
                "  log_every: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "masking:",
                "  token_mask_prob: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    model_cfg = config_dir / "model.yaml"
    model_cfg.write_text(
        "\n".join(
            [
                "model:",
                "  d_model: 32",
                "  n_heads: 4",
                "  n_layers: 1",
                "  d_ffn: 64",
                "  dropout: 0.0",
                "  max_profile_tokens: 4",
                "  max_event_tokens: 4",
                "  max_events: 3",
                "  profile_layers: 1",
                "  event_layers: 1",
                "  history_layers: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_mlm",
            "--config",
            str(train_cfg),
            "--model_config",
            str(model_cfg),
            "--data_dir",
            str(tokenized_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(output_dir),
            "--tokenizer_dir",
            str(tokenizer_dir),
            "--device",
            "cpu",
        ],
    )
    pretrain_main()

    assert (output_dir / "best.ckpt").exists()
