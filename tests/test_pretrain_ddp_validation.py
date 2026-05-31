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
        "event_key_ids": torch.tensor([[[10, 11], [12, 11]]], dtype=torch.long),
        "event_value_ids": torch.ones((1, 2, 2), dtype=torch.long),
        "event_value_pos": torch.zeros((1, 2, 2), dtype=torch.long),
        "event_token_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "event_time": torch.zeros((1, 2), dtype=torch.float32),
        "calendar_features": torch.zeros((1, 2, 6), dtype=torch.float32),
        "event_mask": torch.ones((1, 2), dtype=torch.bool),
        "mlm_labels": torch.ones((1, 2, 2), dtype=torch.long),
        "mlm_value_type_ids": torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long),
        "mlm_source_token_mask": torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
        "mlm_source_key_mask": torch.tensor([[[0, 1], [0, 0]]], dtype=torch.bool),
        "mlm_source_event_mask": torch.tensor([[[0, 0], [1, 1]]], dtype=torch.bool),
    }


def _make_all_ignore_batch() -> dict[str, torch.Tensor]:
    batch = _make_batch()
    batch["mlm_labels"] = torch.full((1, 2, 2), -100, dtype=torch.long)
    return batch


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

    valid_metrics = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        precision="fp32",
    )

    assert called["all_reduce"] == 1
    assert float(valid_metrics["valid_loss"]) > 0.0
    assert float(valid_metrics["valid_masked_accuracy"]) >= 0.0


def test_evaluate_skips_all_reduce_outside_distributed(monkeypatch) -> None:
    def fail_all_reduce(stats: torch.Tensor, op: object | None = None) -> None:
        _ = (stats, op)
        raise AssertionError("all_reduce should not be called when distributed mode is disabled")

    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: False)
    monkeypatch.setattr(pretrain_mlm.dist, "all_reduce", fail_all_reduce)

    model = _DummyModel(vocab_size=4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loader = [_make_batch()]

    valid_metrics = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        precision="fp32",
    )

    assert float(valid_metrics["valid_loss"]) > 0.0
    assert float(valid_metrics["valid_masked_accuracy"]) >= 0.0


def test_evaluate_skips_batches_without_supervised_targets(monkeypatch) -> None:
    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: False)

    model = _DummyModel(vocab_size=4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loader = [_make_all_ignore_batch()]

    valid_metrics = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        precision="fp32",
    )

    assert valid_metrics["valid_batches"] == 0.0
    assert valid_metrics["valid_loss"] == float("inf")
    assert valid_metrics["valid_masked_accuracy"] == 0.0
    assert valid_metrics["valid_top5_acc"] == 0.0


def test_evaluate_emits_stratified_validation_metrics(monkeypatch) -> None:
    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: False)

    model = _DummyModel(vocab_size=16)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loader = [_make_batch()]
    numeric_bucket_lookup = torch.full((16,), -1, dtype=torch.long)
    numeric_bucket_lookup[1] = 4

    valid_metrics = pretrain_mlm._evaluate(
        model=model,
        valid_loader=valid_loader,  # type: ignore[arg-type]
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        precision="fp32",
        token_id_to_numeric_bucket=numeric_bucket_lookup,
        key_id_to_metric_name={
            10: "e_direction",
            11: "e_amount_paid",
            12: "e_description",
        },
    )

    assert valid_metrics["valid_masked_accuracy"] == 1.0
    assert valid_metrics["valid_acc_overall"] == 1.0
    assert valid_metrics["valid_acc_categorical"] == 1.0
    assert valid_metrics["valid_acc_numerical"] == 1.0
    assert valid_metrics["valid_acc_text"] == 1.0
    assert valid_metrics["valid_acc_token_mask"] == 1.0
    assert valid_metrics["valid_acc_key_mask"] == 1.0
    assert valid_metrics["valid_acc_event_mask"] == 1.0
    assert valid_metrics["valid_top5_acc"] == 1.0
    assert valid_metrics["valid_num_bucket_mae"] == 0.0
    assert valid_metrics["valid_num_within_1_acc"] == 1.0
    assert valid_metrics["valid_acc_by_key_e_amount_paid"] == 1.0


def test_supervised_target_rank_count_reduces_across_ranks(monkeypatch) -> None:
    called = {"all_reduce": 0}

    def fake_all_reduce(stats: torch.Tensor, op: object | None = None) -> None:
        _ = op
        called["all_reduce"] += 1
        stats.fill_(1)

    monkeypatch.setattr(pretrain_mlm, "_is_distributed", lambda: True)
    monkeypatch.setattr(pretrain_mlm.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(pretrain_mlm.dist, "ReduceOp", type("_ReduceOp", (), {"SUM": object()})())

    reduced = pretrain_mlm._supervised_target_rank_count(False, torch.device("cpu"))

    assert called["all_reduce"] == 1
    assert reduced == 1
