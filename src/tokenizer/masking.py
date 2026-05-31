from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch

from src.training.data import _trim_stacked_batch


@dataclass
class MaskedEventCollator:
    mask_token_id: int
    unk_token_id: int
    pad_token_id: int = 0
    key_id_to_value_type_id: np.ndarray | None = None
    special_token_ids: tuple[int, ...] = ()
    token_mask_probability: float = 0.15
    event_mask_probability: float = 0.10
    key_mask_probability: float = 0.10
    unk_probability: float = 0.10
    seed: int = 42
    ignore_index: int = -100

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._debug_calls = 0

    def _debug_event(self, event: str, **payload: object) -> None:
        env_path = Path(os.environ.get("PRAGMA_DEBUG_ENV_FILE", ".dbg/pretrain-slow.env"))
        url = "http://127.0.0.1:7777/event"
        session_id = "pretrain-slow"
        run_id = os.environ.get("PRAGMA_DEBUG_RUN_ID", "pre-fix")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1].strip() or url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        body = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": "B",
            "location": "src/tokenizer/masking.py",
            "msg": f"[DEBUG] {event}",
            "data": payload,
            "ts": int(time.time() * 1000),
        }
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=0.25,
            ).read()
        except Exception:
            return

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eligible = (
            event_token_mask.astype(bool)
            & event_mask[:, None].astype(bool)
            & (event_value_ids != self.pad_token_id)
        )
        mlm_mask = np.zeros_like(event_value_ids, dtype=bool)

        token_source_mask = (self.rng.random(event_value_ids.shape) < self.token_mask_probability) & eligible
        mlm_mask |= token_source_mask

        event_source_mask = np.zeros_like(event_value_ids, dtype=bool)
        event_select = self.rng.random(event_mask.shape[0]) < self.event_mask_probability
        for event_idx in np.nonzero(event_select & event_mask.astype(bool))[0]:
            event_source_mask[event_idx] |= eligible[event_idx]
        mlm_mask |= event_source_mask

        key_source_mask = np.zeros_like(event_value_ids, dtype=bool)
        present_keys = np.unique(event_key_ids[eligible])
        if present_keys.size > 0:
            key_select = self.rng.random(present_keys.shape[0]) < self.key_mask_probability
            selected_keys = set(present_keys[key_select].tolist())
            if selected_keys:
                key_source_mask = np.isin(event_key_ids, list(selected_keys)) & eligible
                mlm_mask |= key_source_mask

        if eligible.any() and not mlm_mask.any():
            first_valid = np.argwhere(eligible)[0]
            mlm_mask[first_valid[0], first_valid[1]] = True
            token_source_mask[first_valid[0], first_valid[1]] = True

        unk_mask = (self.rng.random(event_value_ids.shape) < self.unk_probability) & mlm_mask
        return mlm_mask, unk_mask, token_source_mask, key_source_mask, event_source_mask

    def _value_type_ids(self, event_key_ids: np.ndarray, event_value_ids: np.ndarray) -> np.ndarray:
        # 0=categorical, 1=numerical, 2=text, 3=special/other.
        value_type_ids = np.full(event_value_ids.shape, 3, dtype=np.int64)
        if self.key_id_to_value_type_id is not None and self.key_id_to_value_type_id.size > 0:
            clipped_key_ids = np.clip(event_key_ids, 0, self.key_id_to_value_type_id.shape[0] - 1)
            value_type_ids = self.key_id_to_value_type_id[clipped_key_ids].astype(np.int64, copy=False)
        if self.special_token_ids:
            value_type_ids[np.isin(event_value_ids, list(self.special_token_ids))] = 3
        return value_type_ids

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # #region debug-point B:collate-timing
        collate_started_at = time.perf_counter()
        # #endregion
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
            "mlm_value_type_ids": [],
            "mlm_source_token_mask": [],
            "mlm_source_key_mask": [],
            "mlm_source_event_mask": [],
            "unk_mask": [],
            "label": [],
        }

        for record in batch:
            event_key_ids = self._normalize(record["event_key_ids"], np.int64)
            event_value_ids = self._normalize(record["event_value_ids"], np.int64)
            event_token_mask = self._normalize(record["event_token_mask"], np.bool_)
            event_mask = self._normalize(record["event_mask"], np.bool_)

            (
                mlm_mask,
                unk_mask,
                token_source_mask,
                key_source_mask,
                event_source_mask,
            ) = self._sample_mask(event_key_ids, event_value_ids, event_token_mask, event_mask)
            mlm_labels = np.full(event_value_ids.shape, self.ignore_index, dtype=np.int64)
            mlm_labels[mlm_mask] = event_value_ids[mlm_mask]
            mlm_labels[unk_mask] = self.ignore_index
            mlm_value_type_ids = self._value_type_ids(event_key_ids, event_value_ids)

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
            out_rows["mlm_value_type_ids"].append(torch.tensor(mlm_value_type_ids))
            out_rows["mlm_source_token_mask"].append(torch.tensor(token_source_mask))
            out_rows["mlm_source_key_mask"].append(torch.tensor(key_source_mask))
            out_rows["mlm_source_event_mask"].append(torch.tensor(event_source_mask))
            out_rows["unk_mask"].append(torch.tensor(unk_mask))
            out_rows["label"].append(torch.tensor(int(record.get("label", 0)), dtype=torch.long))
        stacked = {key: torch.stack(values, dim=0) for key, values in out_rows.items()}
        stacked = _trim_stacked_batch(stacked)
        # #region debug-point B:collate-timing
        self._debug_calls += 1
        if self._debug_calls <= 5 or self._debug_calls % 50 == 0:
            self._debug_event(
                "collate_timing",
                call_index=self._debug_calls,
                batch_size=len(batch),
                event_shape=list(stacked["event_value_ids"].shape),
                profile_shape=list(stacked["profile_value_ids"].shape),
                elapsed_s=round(time.perf_counter() - collate_started_at, 4),
            )
        # #endregion
        return stacked
