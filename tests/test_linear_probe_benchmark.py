from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from scripts.run_lite_benchmark import main as benchmark_main
from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel
from src.training.checkpoint import save_checkpoint
from src.training.linear_probe import main as linear_probe_main


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


def _write_dataset(data_dir: Path) -> None:
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
    pd.DataFrame(rows).to_parquet(data_dir / "dataset.parquet", index=False)


def _write_splits(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_ids.txt").write_text("1\n2\n3\n4\n", encoding="utf-8")
    (split_dir / "valid_ids.txt").write_text("5\n", encoding="utf-8")
    (split_dir / "test_ids.txt").write_text("6\n", encoding="utf-8")


def _write_checkpoint(ckpt_path: Path, tokenizer_dir: Path) -> None:
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


def test_linear_probe_outputs_metrics_and_lbfgs_probe(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "splits"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "probe"
    data_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_tokenizer(tokenizer_dir)
    _write_dataset(data_dir)
    _write_splits(split_dir)
    ckpt_path = tmp_path / "best.ckpt"
    _write_checkpoint(ckpt_path, tokenizer_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linear_probe",
            "--checkpoint",
            str(ckpt_path),
            "--data_dir",
            str(data_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--repr_type",
            "concat",
            "--batch_size",
            "2",
        ],
    )
    linear_probe_main()

    report = json.loads((output_dir / "concat_metrics.json").read_text(encoding="utf-8"))
    assert report["repr_type"] == "concat"
    assert "valid_metrics" in report
    assert "test_metrics" in report

    with (output_dir / "concat_probe.pkl").open("rb") as f:
        probe = pickle.load(f)
    assert probe["classifier"].solver == "lbfgs"
    assert hasattr(probe["scaler"], "mean_")


def test_run_lite_benchmark_writes_aggregate_report(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "splits"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "benchmark"
    data_dir.mkdir()
    tokenizer_dir.mkdir()
    _write_tokenizer(tokenizer_dir)
    _write_dataset(data_dir)
    _write_splits(split_dir)
    ckpt_path = tmp_path / "best.ckpt"
    _write_checkpoint(ckpt_path, tokenizer_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lite_benchmark",
            "--checkpoint",
            str(ckpt_path),
            "--data_dir",
            str(data_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--num_records",
            "6",
            "--repr_types",
            "zh_usr",
            "concat",
            "--batch_size",
            "2",
        ],
    )
    benchmark_main()

    report = json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["num_records"] == 6
    assert set(report["results"].keys()) == {"zh_usr", "concat"}
    assert (output_dir / "sampled_splits" / "train_ids.txt").exists()
