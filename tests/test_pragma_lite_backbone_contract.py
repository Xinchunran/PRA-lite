from __future__ import annotations

import torch

from src.model.pragma_lite import PragmaLiteModel
from src.model.pragma_lite.model import _canonicalize_grad_layout


def _make_model() -> PragmaLiteModel:
    model = PragmaLiteModel(
        vocab_size=128,
        d_model=32,
        n_heads=4,
        d_ffn=64,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
        dropout=0.0,
        max_profile_tokens=8,
        max_event_tokens=6,
        max_events=4,
    )
    model.eval()
    return model


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "profile_key_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
        "profile_value_ids": torch.tensor([[41, 42, 43]], dtype=torch.long),
        "profile_value_pos": torch.tensor([[0, 0, 0]], dtype=torch.long),
        "profile_time": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "profile_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "event_key_ids": torch.tensor([[[21, 22, 23], [31, 32, 33]]], dtype=torch.long),
        "event_value_ids": torch.tensor([[[51, 52, 53], [61, 62, 63]]], dtype=torch.long),
        "event_value_pos": torch.tensor([[[0, 0, 0], [0, 0, 0]]], dtype=torch.long),
        "event_token_mask": torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.bool),
        "event_time": torch.tensor([[100.0, 10.0]], dtype=torch.float32),
        "calendar_features": torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ),
        "event_mask": torch.tensor([[1, 1]], dtype=torch.bool),
    }


def test_history_encoder_outputs_user_and_event_states() -> None:
    model = _make_model()
    batch = _make_batch()

    with torch.no_grad():
        out = model(**batch)

    assert "history_embedding" in out
    assert "history_event_hidden" in out
    assert "zh_usr" in out and "zh_evt" in out
    assert out["history_embedding"].shape == (1, model.d_model)
    assert out["history_event_hidden"].shape == (1, 2, model.d_model)
    assert out["zh_usr"].shape == (1, model.d_model)
    assert out["zh_evt"].shape == (1, 2, model.d_model)


def test_event_encoder_splits_evt_token_from_local_token_hidden() -> None:
    model = _make_model()
    batch = _make_batch()

    with torch.no_grad():
        out = model(**batch)

    assert out["event_token_hidden"].shape == (1, 2, batch["event_key_ids"].size(-1), model.d_model)
    assert out["event_embeddings"].shape == (1, 2, model.d_model)


def test_history_event_hidden_changes_when_event_order_changes() -> None:
    model = _make_model()
    batch = _make_batch()

    with torch.no_grad():
        base = model(**batch)["history_event_hidden"]

    swapped = dict(batch)
    swapped["event_key_ids"] = batch["event_key_ids"].flip(1)
    swapped["event_value_ids"] = batch["event_value_ids"].flip(1)
    swapped["event_value_pos"] = batch["event_value_pos"].flip(1)
    swapped["event_token_mask"] = batch["event_token_mask"].flip(1)
    swapped["event_time"] = batch["event_time"].flip(1)
    swapped["calendar_features"] = batch["calendar_features"].flip(1)
    swapped["event_mask"] = batch["event_mask"].flip(1)
    with torch.no_grad():
        changed = model(**swapped)["history_event_hidden"]

    assert not torch.allclose(base, changed, atol=1e-6)


def test_mlm_event_context_comes_from_history_encoder_not_pre_history_pooling() -> None:
    model = _make_model()
    batch = _make_batch()

    with torch.no_grad():
        out = model(**batch)
        logits_from_history = model._mlm_logits(
            out["event_token_hidden"].reshape(1, -1, model.d_model),
            out["history_event_hidden"]
            .unsqueeze(2)
            .expand(-1, -1, out["event_token_hidden"].size(2), -1)
            .reshape(1, -1, model.d_model),
            out["history_embedding"],
        )
        logits_from_pre_history = model._mlm_logits(
            out["event_token_hidden"].reshape(1, -1, model.d_model),
            out["event_embeddings"]
            .unsqueeze(2)
            .expand(-1, -1, out["event_token_hidden"].size(2), -1)
            .reshape(1, -1, model.d_model),
            out["history_embedding"],
        )

    assert logits_from_history.shape == logits_from_pre_history.shape
    assert not torch.allclose(logits_from_history, logits_from_pre_history, atol=1e-6)


def test_canonicalize_grad_layout_rewrites_singleton_stride_to_match_parameter() -> None:
    reference = torch.zeros((1, 1, 32), dtype=torch.float32)
    grad = torch.empty_strided((1, 1, 32), (96, 32, 1), dtype=torch.float32)
    fixed = _canonicalize_grad_layout(grad, reference)

    assert tuple(grad.stride()) == (96, 32, 1)
    assert tuple(reference.stride()) == (32, 32, 1)
    assert tuple(fixed.stride()) == tuple(reference.stride())
