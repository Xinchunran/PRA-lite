from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _normalize_nested_array(value: object, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=dtype)


def read_ids(path: str | Path) -> set[int]:
    p = Path(path)
    return {int(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip() != ""}


@dataclass(frozen=True)
class Batch:
    entity_id: torch.Tensor
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_value_pos: torch.Tensor
    profile_time: torch.Tensor
    profile_mask: torch.Tensor
    event_key_ids: torch.Tensor
    event_value_ids: torch.Tensor
    event_value_pos: torch.Tensor
    event_token_mask: torch.Tensor
    event_time: torch.Tensor
    calendar_features: torch.Tensor
    event_mask: torch.Tensor
    labels: torch.Tensor | None = None

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "profile_key_ids": self.profile_key_ids,
            "profile_value_ids": self.profile_value_ids,
            "profile_value_pos": self.profile_value_pos,
            "profile_time": self.profile_time,
            "profile_mask": self.profile_mask,
            "event_key_ids": self.event_key_ids,
            "event_value_ids": self.event_value_ids,
            "event_value_pos": self.event_value_pos,
            "event_token_mask": self.event_token_mask,
            "event_time": self.event_time,
            "calendar_features": self.calendar_features,
            "event_mask": self.event_mask,
        }


class TokenizedDataset(Dataset):
    def __init__(self, data_path: Path, entity_ids: set[int] | None = None) -> None:
        df = pd.read_parquet(data_path)
        if entity_ids is not None:
            df = df[df["entity_id"].isin(list(entity_ids))]
        required_columns = {
            "profile_key_ids",
            "profile_value_ids",
            "profile_value_pos",
            "profile_time",
            "profile_mask",
            "event_key_ids",
            "event_value_ids",
            "event_value_pos",
            "event_token_mask",
            "event_time",
            "calendar_features",
            "event_mask",
        }
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                "Tokenized dataset must contain structured PRAGMA columns only; missing: "
                + ", ".join(sorted(missing))
            )
        self.entity_id = df["entity_id"].astype("int64").to_numpy()
        self.profile_key_ids = df["profile_key_ids"].tolist()
        self.profile_value_ids = df["profile_value_ids"].tolist()
        self.profile_value_pos = df["profile_value_pos"].tolist()
        self.profile_time = df["profile_time"].tolist()
        self.profile_mask = df["profile_mask"].tolist()
        self.event_key_ids = df["event_key_ids"].tolist()
        self.event_value_ids = df["event_value_ids"].tolist()
        self.event_value_pos = df["event_value_pos"].tolist()
        self.event_token_mask = df["event_token_mask"].tolist()
        self.event_time = df["event_time"].tolist()
        self.calendar_features = df["calendar_features"].tolist()
        self.event_mask = df["event_mask"].tolist()
        self.label = df["label"].astype("int64").to_numpy() if "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.entity_id)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "entity_id": int(self.entity_id[idx]),
            "profile_key_ids": self.profile_key_ids[idx],
            "profile_value_ids": self.profile_value_ids[idx],
            "profile_value_pos": self.profile_value_pos[idx],
            "profile_time": self.profile_time[idx],
            "profile_mask": self.profile_mask[idx],
            "event_key_ids": self.event_key_ids[idx],
            "event_value_ids": self.event_value_ids[idx],
            "event_value_pos": self.event_value_pos[idx],
            "event_token_mask": self.event_token_mask[idx],
            "event_time": self.event_time[idx],
            "calendar_features": self.calendar_features[idx],
            "event_mask": self.event_mask[idx],
        }
        if self.label is not None:
            item["label"] = int(self.label[idx])
        return item


def pad_collate(batch: list[dict], pad_id: int) -> Batch:
    _ = pad_id
    entity_id = torch.tensor([x["entity_id"] for x in batch], dtype=torch.long)
    profile_key_ids = torch.as_tensor(
        np.stack([_normalize_nested_array(x["profile_key_ids"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    profile_value_ids = torch.as_tensor(
        np.stack([_normalize_nested_array(x["profile_value_ids"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    profile_value_pos = torch.as_tensor(
        np.stack([_normalize_nested_array(x["profile_value_pos"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    profile_time = torch.as_tensor(
        np.stack([_normalize_nested_array(x["profile_time"], np.float32) for x in batch], axis=0),
        dtype=torch.float32,
    )
    profile_mask = torch.as_tensor(
        np.stack([_normalize_nested_array(x["profile_mask"], np.bool_) for x in batch], axis=0),
        dtype=torch.bool,
    )
    event_key_ids = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_key_ids"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    event_value_ids = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_value_ids"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    event_value_pos = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_value_pos"], np.int64) for x in batch], axis=0),
        dtype=torch.long,
    )
    event_token_mask = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_token_mask"], np.bool_) for x in batch], axis=0),
        dtype=torch.bool,
    )
    event_time = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_time"], np.float32) for x in batch], axis=0),
        dtype=torch.float32,
    )
    calendar_features = torch.as_tensor(
        np.stack([_normalize_nested_array(x["calendar_features"], np.float32) for x in batch], axis=0),
        dtype=torch.float32,
    )
    event_mask = torch.as_tensor(
        np.stack([_normalize_nested_array(x["event_mask"], np.bool_) for x in batch], axis=0),
        dtype=torch.bool,
    )
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32) if "label" in batch[0] else None
    return Batch(
        entity_id=entity_id,
        profile_key_ids=profile_key_ids,
        profile_value_ids=profile_value_ids,
        profile_value_pos=profile_value_pos,
        profile_time=profile_time,
        profile_mask=profile_mask,
        event_key_ids=event_key_ids,
        event_value_ids=event_value_ids,
        event_value_pos=event_value_pos,
        event_token_mask=event_token_mask,
        event_time=event_time,
        calendar_features=calendar_features,
        event_mask=event_mask,
        labels=labels,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
