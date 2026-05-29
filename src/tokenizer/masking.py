from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MaskedEventCollator:
    mask_token_id: int
    unk_token_id: int
    pad_token_id: int = 0
    token_mask_probability: float = 0.15
    event_mask_probability: float = 0.10
    key_mask_probability: float = 0.10
    unk_probability: float = 0.10
    seed: int = 42
    ignore_index: int = -100

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _normalize(self, value: object, dtype: np.dtype) -> np.ndarray:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return np.asarray(value, dtype=dtype)

    def _sample_mask(
        self,
        event_key_ids: np.ndarray,
        event_value_ids: np.ndarray,
        event_token_mask: np.ndarray,
        event_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eligible = (
            event_token_mask.astype(bool)
            & event_mask[:, None].astype(bool)
            & (event_value_ids != self.pad_token_id)
        )
        mlm_mask = np.zeros_like(event_value_ids, dtype=bool)

        token_mask = self.rng.random(event_value_ids.shape) < self.token_mask_probability
        mlm_mask |= token_mask & eligible

        event_select = self.rng.random(event_mask.shape[0]) < self.event_mask_probability
        for event_idx in np.nonzero(event_select & event_mask.astype(bool))[0]:
            mlm_mask[event_idx] |= eligible[event_idx]

        present_keys = np.unique(event_key_ids[eligible])
        if present_keys.size > 0:
            key_select = self.rng.random(present_keys.shape[0]) < self.key_mask_probability
            selected_keys = set(present_keys[key_select].tolist())
            if selected_keys:
                mlm_mask |= np.isin(event_key_ids, list(selected_keys)) & eligible

        if eligible.any() and not mlm_mask.any():
            first_valid = np.argwhere(eligible)[0]
            mlm_mask[first_valid[0], first_valid[1]] = True

        unk_mask = (self.rng.random(event_value_ids.shape) < self.unk_probability) & mlm_mask
        return mlm_mask, unk_mask

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        out_rows: dict[str, list[torch.Tensor]] = {
            "entity_id": [],
            "profile_key_ids": [],
            "profile_value_ids": [],
            "profile_value_pos": [],
            "profile_time": [],
            "profile_mask": [],
            "event_key_ids": [],
            "event_value_ids": [],
            "event_value_pos": [],
            "event_token_mask": [],
            "event_time": [],
            "calendar_features": [],
            "event_mask": [],
            "mlm_labels": [],
            "unk_mask": [],
            "label": [],
        }

        for record in batch:
            event_key_ids = self._normalize(record["event_key_ids"], np.int64)
            event_value_ids = self._normalize(record["event_value_ids"], np.int64)
            event_token_mask = self._normalize(record["event_token_mask"], np.bool_)
            event_mask = self._normalize(record["event_mask"], np.bool_)

            mlm_mask, unk_mask = self._sample_mask(event_key_ids, event_value_ids, event_token_mask, event_mask)
            mlm_labels = np.full(event_value_ids.shape, self.ignore_index, dtype=np.int64)
            mlm_labels[mlm_mask] = event_value_ids[mlm_mask]
            mlm_labels[unk_mask] = self.ignore_index

            masked_values = event_value_ids.copy()
            masked_values[mlm_mask] = self.mask_token_id
            masked_values[unk_mask] = self.unk_token_id

            out_rows["entity_id"].append(torch.tensor(int(record["entity_id"]), dtype=torch.long))
            out_rows["profile_key_ids"].append(torch.tensor(self._normalize(record["profile_key_ids"], np.int64)))
            out_rows["profile_value_ids"].append(torch.tensor(self._normalize(record["profile_value_ids"], np.int64)))
            out_rows["profile_value_pos"].append(torch.tensor(self._normalize(record["profile_value_pos"], np.int64)))
            out_rows["profile_time"].append(torch.tensor(self._normalize(record["profile_time"], np.float32)))
            out_rows["profile_mask"].append(torch.tensor(self._normalize(record["profile_mask"], np.bool_)))
            out_rows["event_key_ids"].append(torch.tensor(event_key_ids))
            out_rows["event_value_ids"].append(torch.tensor(masked_values))
            out_rows["event_value_pos"].append(torch.tensor(self._normalize(record["event_value_pos"], np.int64)))
            out_rows["event_token_mask"].append(torch.tensor(event_token_mask))
            out_rows["event_time"].append(torch.tensor(self._normalize(record["event_time"], np.float32)))
            out_rows["calendar_features"].append(torch.tensor(self._normalize(record["calendar_features"], np.float32)))
            out_rows["event_mask"].append(torch.tensor(event_mask))
            out_rows["mlm_labels"].append(torch.tensor(mlm_labels))
            out_rows["unk_mask"].append(torch.tensor(unk_mask))
            out_rows["label"].append(torch.tensor(int(record.get("label", 0)), dtype=torch.long))

        return {key: torch.stack(values, dim=0) for key, values in out_rows.items()}
