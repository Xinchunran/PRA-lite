from __future__ import annotations

import torch

from src.training import pretrain_mlm


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, *, event_value_ids: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        batch, events, tokens = event_value_ids.shape
        logits = torch.zeros((batch, events, tokens, self.vocab_size), dtype=torch.float32)
        logits[..., 1] = 1.0
        return logits


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "profile_key_ids": torch.zeros((1, 2), dtype=torch.long),
        "profile_value_ids": torch.zeros((1, 2), dtype=torch.long),
        "profile_value_pos": torch.zeros((1, 2), dtype=torch.long),
        "profile_time": torch.zeros((1, 2), dtype=torch.float32),
        "profile_mask": torch.ones((1, 2), dtype=torch.bool),
        "event_key_ids": torch.zeros((1, 2, 2), dtype=torch.long),
        "event_value_ids": torch.ones((1, 2, 2), dtype=torch.long),
        "event_value_pos": torch.zeros((1, 2, 2), dtype=torch.long),
        "event_token_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "event_time": torch.zeros((1, 2), dtype=torch.float32),
        "calendar_features": torch.zeros((1, 2, 6), dtype=torch.float32),
        "event_mask": torch.ones((1, 2), dtype=torch.bool),
        "mlm_labels": torch.ones((1, 2, 2), dtype=torch.long),
    }


def test_evaluate_reduces_validation_loss_across_ranks(monkeypatch) -> None:
    called = {"all_reduce": 0}

    def fake_all_reduce(stats: torch.Tensor, op: object | None = None) -> None:
        _ = op
        called["all_reduce"] += 1
        stats.mul_(2.0)

    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: True)
    monkeypatch.setattr(pretrain_mlm.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(pretrain_mlm.dist, "ReduceOp", type("_ReduceOp", (), {"SUM": object()})())

    model = _DummyModel(vocab_size=4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loader = [_make_batch(), _make_batch()]

    valid_loss = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    assert called["all_reduce"] == 1
    assert valid_loss > 0.0


def test_evaluate_skips_all_reduce_outside_distributed(monkeypatch) -> None:
    def fail_all_reduce(stats: torch.Tensor, op: object | None = None) -> None:
        _ = (stats, op)
        raise AssertionError("all_reduce should not be called when distributed mode is disabled")

    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: False)
    monkeypatch.setattr(pretrain_mlm.dist, "all_reduce", fail_all_reduce)

    model = _DummyModel(vocab_size=4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loader = [_make_batch()]

    valid_loss = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    assert valid_loss > 0.0
