from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.training import pretrain_mlm
from tests.test_structured_data_pipeline import _write_tokenizer
from tools.plot_pretrain_metrics import DEFAULT_PLOT_FILES, _segmented_series, generate_plots, load_metrics


def _write_metrics(metrics_path: Path) -> None:
    entries = [
        {"kind": "train", "step": 100, "train_loss": 4.2, "masked_accuracy": 0.21, "grad_norm": 1.5, "learning_rate": 1e-4, "steps_per_sec": 3.0, "samples_per_sec": 48.0, "tokens_per_sec": 960.0, "data_wait_s": 0.01, "h2d_s": 0.001, "forward_s": 0.12, "backward_s": 0.16, "optimizer_s": 0.04, "total_step_s": 0.33, "gpu_mem_allocated_gb": 6.4, "gpu_mem_reserved_gb": 8.0, "num_ready_shards": 8},
        {"kind": "train", "step": 5000, "train_loss": 2.1, "masked_accuracy": 0.55, "grad_norm": 0.9, "learning_rate": 8e-5, "steps_per_sec": 3.1, "samples_per_sec": 49.6, "tokens_per_sec": 992.0, "data_wait_s": 0.02, "h2d_s": 0.001, "forward_s": 0.11, "backward_s": 0.15, "optimizer_s": 0.04, "total_step_s": 0.32, "gpu_mem_allocated_gb": 6.5, "gpu_mem_reserved_gb": 8.1, "num_ready_shards": 16},
        {"kind": "valid", "step": 1000, "valid_loss": 2.5, "valid_masked_accuracy": 0.42, "valid_perplexity": 12.2, "valid_batches": 32, "best_valid_loss": 2.5, "num_ready_shards": 10},
        {"kind": "valid", "step": 5000, "valid_loss": 1.8, "valid_masked_accuracy": 0.64, "valid_perplexity": 6.1, "valid_batches": 64, "best_valid_loss": 1.8, "num_ready_shards": 16},
    ]
    metrics_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _write_tokenized_splits(tokenized_dir: Path) -> None:
    train_df = pd.DataFrame(
        [
            {
                "entity_id": 1,
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
                "entity_id": 2,
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
                "entity_id": 3,
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
            }
        ]
    )
    train_df.to_parquet(tokenized_dir / "train.parquet", index=False, row_group_size=1)
    valid_df.to_parquet(tokenized_dir / "valid.parquet", index=False, row_group_size=1)


def test_generate_plots_writes_train_and_validation_outputs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    output_dir = tmp_path / "plots"
    _write_metrics(metrics_path)

    entries = load_metrics(metrics_path)
    generate_plots(entries, output_dir, "Unit Test")

    missing = [name for name in DEFAULT_PLOT_FILES if not (output_dir / name).exists()]
    assert missing == []


def test_segmented_series_splits_on_step_reset(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "train", "step": 1, "train_loss": 10.0}),
                json.dumps({"kind": "train", "step": 2, "train_loss": 8.0}),
                json.dumps({"kind": "train", "step": 1, "train_loss": 9.5}),
                json.dumps({"kind": "train", "step": 2, "train_loss": 7.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_metrics(metrics_path)
    segments = _segmented_series(entries, "train", "train_loss")

    assert segments == [([1, 2], [10.0, 8.0]), ([1, 2], [9.5, 7.5])]


def test_pretrain_periodic_plotting_invokes_plot_writer(tmp_path: Path, monkeypatch) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenizer_dir = tmp_path / "tokenizer"
    split_dir = tmp_path / "splits"
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "runs"
    plots_dir = tmp_path / "plots"
    tokenized_dir.mkdir()
    tokenizer_dir.mkdir()
    split_dir.mkdir()
    config_dir.mkdir()
    _write_tokenizer(tokenizer_dir)
    _write_tokenized_splits(tokenized_dir)

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
                "  valid_every: 1",
                "  full_valid_every: 1",
                "  plot_every: 1",
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

    calls: list[tuple[Path, Path, str]] = []

    def fake_plot(metrics_path: Path, output_path: Path, title_prefix: str) -> None:
        calls.append((metrics_path, output_path, title_prefix))

    monkeypatch.setattr(pretrain_mlm, "_maybe_generate_plots", fake_plot)
    monkeypatch.setenv("PLOTS_DIR", str(plots_dir))
    monkeypatch.setenv("PLOT_TITLE_PREFIX", "UnitTest")
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

    pretrain_mlm.main()

    assert calls
    assert calls[0][0] == output_dir / "metrics.jsonl"
    assert calls[0][1] == plots_dir
    assert calls[0][2] == "UnitTest"


def test_quick_validation_writes_stratified_metrics_to_jsonl(tmp_path: Path, monkeypatch) -> None:
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
    _write_tokenized_splits(tokenized_dir)

    train_cfg = config_dir / "pretrain_quick_valid.yaml"
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
                "  valid_every: 1",
                "  full_valid_every: 10",
                "  max_valid_batches: 1",
                "  plot_every: 0",
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

    pretrain_mlm.main()

    entries = load_metrics(output_dir / "metrics.jsonl")
    valid_entries = [entry for entry in entries if entry.get("kind") == "valid"]
    assert len(valid_entries) == 1
    valid_entry = valid_entries[0]
    assert valid_entry["eval_mode"] == "quick"
    assert "valid_acc_categorical" in valid_entry
    assert "valid_acc_numerical" in valid_entry
    assert "valid_acc_text" in valid_entry
    assert "valid_top5_acc" in valid_entry
