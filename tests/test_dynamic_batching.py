from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.training.data import DistributedTokenBudgetBatchSampler, _trim_stacked_batch


class _MetaDataset(Dataset):
    def __init__(self, stats: list[tuple[int, int]]) -> None:
        self.stats = stats

    def __len__(self) -> int:
        return len(self.stats)

    def __getitem__(self, idx: int) -> dict[str, int]:
        event_count, profile_tokens = self.stats[idx]
        return {
            "batching_event_count": event_count,
            "batching_profile_token_count": profile_tokens,
        }

    def get_batching_stats(self, idx: int) -> tuple[int, int]:
        return self.stats[idx]


def test_token_budget_batch_sampler_respects_budget_and_batch_cap() -> None:
    dataset = _MetaDataset([(4, 2), (4, 2), (20, 2), (2, 2)])
    sampler = DistributedTokenBudgetBatchSampler(
        dataset,
        token_budget=40,
        max_event_tokens=4,
        max_batch_size=2,
        bucket_boundaries=[8, 16, 32],
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    batches = list(iter(sampler))

    assert batches == [[0, 1], [3], [2]]
    for batch in batches:
        total_cost = sum(dataset.stats[idx][0] * 4 + dataset.stats[idx][1] for idx in batch)
        assert len(batch) <= 2
        assert total_cost <= 40 or len(batch) == 1


def test_token_budget_batch_sampler_balances_batch_count_across_ranks() -> None:
    dataset = _MetaDataset([(2, 2), (2, 2), (2, 2), (9, 2), (9, 2)])
    rank0 = DistributedTokenBudgetBatchSampler(
        dataset,
        token_budget=32,
        max_event_tokens=4,
        max_batch_size=2,
        bucket_boundaries=[8, 16, 32],
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    rank1 = DistributedTokenBudgetBatchSampler(
        dataset,
        token_budget=32,
        max_event_tokens=4,
        max_batch_size=2,
        bucket_boundaries=[8, 16, 32],
        shuffle=False,
        num_replicas=2,
        rank=1,
    )

    batches0 = list(iter(rank0))
    batches1 = list(iter(rank1))

    assert len(batches0) == len(batches1)
    covered = sorted({idx for batch in batches0 + batches1 for idx in batch})
    assert covered == [0, 1, 2, 3, 4]


def test_trim_stacked_batch_crops_to_batch_max_active_lengths() -> None:
    batch = {
        "entity_id": torch.tensor([1, 2], dtype=torch.long),
        "profile_key_ids": torch.zeros((2, 4), dtype=torch.long),
        "profile_value_ids": torch.zeros((2, 4), dtype=torch.long),
        "profile_value_pos": torch.zeros((2, 4), dtype=torch.long),
        "profile_time": torch.zeros((2, 4), dtype=torch.float32),
        "profile_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool),
        "event_key_ids": torch.zeros((2, 3, 4), dtype=torch.long),
        "event_value_ids": torch.zeros((2, 3, 4), dtype=torch.long),
        "event_value_pos": torch.zeros((2, 3, 4), dtype=torch.long),
        "event_token_mask": torch.tensor(
            [
                [[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        "event_time": torch.zeros((2, 3), dtype=torch.float32),
        "calendar_features": torch.zeros((2, 3, 6), dtype=torch.float32),
        "event_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        "mlm_labels": torch.zeros((2, 3, 4), dtype=torch.long),
        "unk_mask": torch.zeros((2, 3, 4), dtype=torch.bool),
    }

    trimmed = _trim_stacked_batch(batch)

    assert trimmed["profile_key_ids"].shape == (2, 2)
    assert trimmed["event_key_ids"].shape == (2, 2, 3)
    assert trimmed["mlm_labels"].shape == (2, 2, 3)
