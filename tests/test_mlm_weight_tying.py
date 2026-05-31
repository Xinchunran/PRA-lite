from __future__ import annotations

import torch
from torch.nn import functional as F

from src.model.pragma_lite import PragmaLiteConfig, PragmaLiteModel


def _make_batch() -> dict[str, torch.Tensor]:
    return {
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


def test_mlm_logits_shape_with_tied_embeddings() -> None:
    cfg = PragmaLiteConfig(vocab_size=64, d_model=32, n_heads=4, d_ffn=64, n_layers=1, profile_layers=1, event_layers=1, history_layers=1, dropout=0.0, max_profile_tokens=4, max_event_tokens=4, max_events=3, tie_mlm_to_embedding=True)
    model = PragmaLiteModel(cfg)

    logits = model(**_make_batch(), return_mlm_logits=True)

    assert logits.shape == (1, 3, 4, cfg.vocab_size)


def test_mlm_uses_shared_embedding_weight_when_tied() -> None:
    cfg = PragmaLiteConfig(vocab_size=64, d_model=32, n_heads=4, d_ffn=64, n_layers=1, profile_layers=1, event_layers=1, history_layers=1, dropout=0.0, max_profile_tokens=4, max_event_tokens=4, max_events=3, tie_mlm_to_embedding=True)
    model = PragmaLiteModel(cfg)

    assert model.cfg.tie_mlm_to_embedding is True
    assert model.mlm_bias is not None
    assert model.mlm_out is None
    assert model.kv_embedding.token_emb.weight.shape[0] == model.vocab_size


def test_mlm_backward_with_tied_embeddings_updates_shared_embedding_grad() -> None:
    cfg = PragmaLiteConfig(vocab_size=64, d_model=32, n_heads=4, d_ffn=64, n_layers=1, profile_layers=1, event_layers=1, history_layers=1, dropout=0.0, max_profile_tokens=4, max_event_tokens=4, max_events=3, tie_mlm_to_embedding=True)
    model = PragmaLiteModel(cfg)
    batch = _make_batch()
    labels = torch.full((1, 3, 4), -100, dtype=torch.long)
    labels[0, 0, 0] = 32
    labels[0, 1, 1] = 43

    logits = model(**batch, return_mlm_logits=True)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1), ignore_index=-100)
    loss.backward()

    assert model.kv_embedding.token_emb.weight.grad is not None
