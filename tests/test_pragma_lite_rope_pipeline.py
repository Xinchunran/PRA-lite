from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel
from src.training.checkpoint import load_checkpoint
from src.training.pretrain_mlm import main as pretrain_main


def test_rope_is_enabled_for_profile_and_history_encoders() -> None:
    model = PragmaLiteModel(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ffn=64,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
        dropout=0.0,
        max_profile_tokens=6,
        max_event_tokens=6,
        max_events=4,
    )

    assert model.profile_encoder.use_rope is True
    assert model.history_encoder.use_rope is True
    assert model.event_encoder.use_rope is False


def test_structured_backbone_returns_mlm_logits() -> None:
    cfg = PragmaLiteConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ffn=64,
        n_layers=1,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
        dropout=0.0,
        max_profile_tokens=4,
        max_event_tokens=4,
        max_events=3,
    )
    model = PragmaLiteModel(cfg)
    batch = {
        "profile_key_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "profile_value_ids": torch.tensor([[20, 21, 0, 0]], dtype=torch.long),
        "profile_value_pos": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "profile_time": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "profile_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "event_key_ids": torch.tensor([[[30, 31, 0, 0], [40, 41, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_ids": torch.tensor([[[32, 33, 0, 0], [42, 43, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_pos": torch.zeros((1, 3, 4), dtype=torch.long),
        "event_token_mask": torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool),
        "event_time": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
        "calendar_features": torch.tensor([[[0.0] * 6, [1.0] * 6, [0.0] * 6]], dtype=torch.float32),
        "event_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }

    logits = model(**batch, return_mlm_logits=True)
    hidden = model(**batch, return_mlm_logits=False)
    assert logits.shape == (1, 3, 4, cfg.vocab_size)
    assert hidden["record_embedding"].shape == (1, cfg.d_model)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(hidden["record_embedding"]).all()


def test_structured_backbone_runs_with_time_and_order_ablation_flags_disabled() -> None:
    cfg = PragmaLiteConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ffn=64,
        n_layers=1,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
        dropout=0.0,
        max_profile_tokens=4,
        max_event_tokens=4,
        max_events=3,
        use_additive_time_proj=False,
        use_history_order_emb=False,
    )
    model = PragmaLiteModel(cfg)
    batch = {
        "profile_key_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "profile_value_ids": torch.tensor([[20, 21, 0, 0]], dtype=torch.long),
        "profile_value_pos": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "profile_time": torch.tensor([[5.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        "profile_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "event_key_ids": torch.tensor([[[30, 31, 0, 0], [40, 41, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_ids": torch.tensor([[[32, 33, 0, 0], [42, 43, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_pos": torch.zeros((1, 3, 4), dtype=torch.long),
        "event_token_mask": torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool),
        "event_time": torch.tensor([[6.0, 2.0, 0.0]], dtype=torch.float32),
        "calendar_features": torch.tensor([[[0.0] * 6, [1.0] * 6, [0.0] * 6]], dtype=torch.float32),
        "event_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }

    logits = model(**batch, return_mlm_logits=True)
    assert logits.shape == (1, 3, 4, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_pretrain_pipeline_smoke_runs_with_rope_hierarchical_model(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    tokenized_dir = data_dir / "processed" / "toy" / "tokenized"
    tokenizer_dir = data_dir / "processed" / "toy" / "tokenizer"
    split_dir = data_dir / "splits" / "toy"
    output_dir = tmp_path / "runs"
    config_dir.mkdir(parents=True)
    tokenized_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

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
                "  max_seq_len: 32",
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

    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[USR]": 3,
        "[EVT]": 4,
        "P:a": 5,
        "P:b": 6,
        "E:a": 7,
        "E:b": 8,
        "E:c": 9,
        "E:d": 10,
    }
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "token_to_id": token_to_id,
        "profile_cols": ["region"],
        "event_cols": ["merchant"],
                "numeric_binners": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(
        [
            {
                "entity_id": 1,
                "profile_key_ids": [5, 0, 0, 0],
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[7, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[8, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]],
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
                "profile_value_ids": [6, 0, 0, 0],
                "profile_value_pos": [0, 0, 0, 0],
                "profile_time": [0.0, 0.0, 0.0, 0.0],
                "profile_mask": [1, 0, 0, 0],
                "event_key_ids": [[7, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[8, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                "event_time": [1.8, 0.9, 0.0],
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
                "event_key_ids": [[7, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_ids": [[8, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]],
                "event_value_pos": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "event_token_mask": [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                "event_time": [1.2, 0.0, 0.0],
                "calendar_features": [[0.2] * 6, [0.0] * 6, [0.0] * 6],
                "event_mask": [1, 0, 0],
                "label": 0,
            },
        ]
    )
    df.to_parquet(tokenized_dir / "dataset.parquet", index=False)
    (split_dir / "train_ids.txt").write_text("1\n2\n", encoding="utf-8")
    (split_dir / "valid_ids.txt").write_text("3\n", encoding="utf-8")

    argv = [
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
    ]
    monkeypatch.setattr(sys, "argv", argv)
    pretrain_main()

    ckpt_path = output_dir / "best.ckpt"
    assert ckpt_path.exists()

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = PragmaLiteModel(PragmaLiteConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    batch = {
        "profile_key_ids": torch.tensor([[5, 0, 0, 0]], dtype=torch.long),
        "profile_value_ids": torch.tensor([[6, 0, 0, 0]], dtype=torch.long),
        "profile_value_pos": torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        "profile_time": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "profile_mask": torch.tensor([[1, 0, 0, 0]], dtype=torch.bool),
        "event_key_ids": torch.tensor([[[7, 0, 0, 0], [9, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_ids": torch.tensor([[[8, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "event_value_pos": torch.zeros((1, 3, 4), dtype=torch.long),
        "event_token_mask": torch.tensor([[[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool),
        "event_time": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
        "calendar_features": torch.tensor([[[0.0] * 6, [1.0] * 6, [0.0] * 6]], dtype=torch.float32),
        "event_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }
    with torch.no_grad():
        logits = model(**batch, return_mlm_logits=True)
        hidden = model(**batch)

    assert logits.shape == (1, 3, 4, model.vocab_size)
    assert hidden["record_embedding"].shape == (1, model.d_model)
