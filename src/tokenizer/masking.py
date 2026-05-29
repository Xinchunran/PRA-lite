from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MaskedEventCollator:
    mask_token_id: int
    unk_token_id: int
    mlm_probability: float = 0.15
    seed: int = 42
    ignore_index: int = -100

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _flatten_tokens(self, record: dict) -> np.ndarray:
        profile_tokens = np.asarray(record.get("profile_tokens", []), dtype=np.int64).reshape(-1)
        event_tokens = np.asarray(record.get("event_tokens", []), dtype=np.int64).reshape(-1)
        return np.concatenate([profile_tokens, event_tokens], axis=0)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_rows = []
        label_rows = []
        downstream_labels = []

        for record in batch:
            tokens = self._flatten_tokens(record)
            mlm_labels = np.full(tokens.shape, self.ignore_index, dtype=np.int64)

            if tokens.size > 0:
                mask = self.rng.random(tokens.shape[0]) < self.mlm_probability
                if not np.any(mask):
                    mask[self.rng.integers(0, tokens.shape[0])] = True
                mlm_labels[mask] = tokens[mask]
                tokens = tokens.copy()
                tokens[mask] = self.mask_token_id

            input_rows.append(torch.tensor(tokens, dtype=torch.long))
            label_rows.append(torch.tensor(mlm_labels, dtype=torch.long))
            downstream_labels.append(int(record.get("label", 0)))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_rows, batch_first=True, padding_value=0)
        mlm_labels = torch.nn.utils.rnn.pad_sequence(
            label_rows, batch_first=True, padding_value=self.ignore_index
        )
        attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "label": torch.tensor(downstream_labels, dtype=torch.long),
        }
