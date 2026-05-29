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
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    profile_input_ids: torch.Tensor | None = None
    profile_attention_mask: torch.Tensor | None = None
    event_input_ids: torch.Tensor | None = None
    event_attention_mask: torch.Tensor | None = None
    event_times: torch.Tensor | None = None
    calendar_features: torch.Tensor | None = None
    event_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    @property
    def has_structured_inputs(self) -> bool:
        return self.profile_input_ids is not None and self.event_input_ids is not None

    def model_inputs(self) -> dict[str, torch.Tensor]:
        if self.has_structured_inputs:
            out = {
                "profile_input_ids": self.profile_input_ids,
                "event_input_ids": self.event_input_ids,
                "profile_attention_mask": self.profile_attention_mask,
                "event_attention_mask": self.event_attention_mask,
            }
            if self.event_times is not None:
                out["event_times"] = self.event_times
            if self.calendar_features is not None:
                out["calendar_features"] = self.calendar_features
            return out
        if self.input_ids is None:
            raise ValueError("Batch has neither structured inputs nor flat input_ids")
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


class TokenizedDataset(Dataset):
    def __init__(self, data_path: Path, entity_ids: set[int] | None = None) -> None:
        df = pd.read_parquet(data_path)
        if entity_ids is not None:
            df = df[df["entity_id"].isin(list(entity_ids))]
        self.has_structured_inputs = {
            "profile_input_ids",
            "profile_attention_mask",
            "event_input_ids",
            "event_attention_mask",
        }.issubset(df.columns)
        self.entity_id = df["entity_id"].astype("int64").to_numpy()
        self.input_ids = df["input_ids"].tolist() if "input_ids" in df.columns else None
        self.attention_mask = df["attention_mask"].tolist() if "attention_mask" in df.columns else None
        self.profile_input_ids = df["profile_input_ids"].tolist() if "profile_input_ids" in df.columns else None
        self.profile_attention_mask = (
            df["profile_attention_mask"].tolist() if "profile_attention_mask" in df.columns else None
        )
        self.event_input_ids = df["event_input_ids"].tolist() if "event_input_ids" in df.columns else None
        self.event_attention_mask = (
            df["event_attention_mask"].tolist() if "event_attention_mask" in df.columns else None
        )
        self.event_times = df["event_times"].tolist() if "event_times" in df.columns else None
        self.calendar_features = df["calendar_features"].tolist() if "calendar_features" in df.columns else None
        self.event_mask = df["event_mask"].tolist() if "event_mask" in df.columns else None
        self.label = df["label"].astype("int64").to_numpy() if "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.entity_id)

    def __getitem__(self, idx: int) -> dict:
        item = {"entity_id": int(self.entity_id[idx])}
        if self.input_ids is not None:
            item["input_ids"] = self.input_ids[idx]
            item["attention_mask"] = self.attention_mask[idx]
        if self.has_structured_inputs:
            item["profile_input_ids"] = self.profile_input_ids[idx]
            item["profile_attention_mask"] = self.profile_attention_mask[idx]
            item["event_input_ids"] = self.event_input_ids[idx]
            item["event_attention_mask"] = self.event_attention_mask[idx]
            item["event_times"] = self.event_times[idx]
            item["calendar_features"] = self.calendar_features[idx]
            item["event_mask"] = self.event_mask[idx]
        if self.label is not None:
            item["label"] = int(self.label[idx])
        return item


def pad_collate(batch: list[dict], pad_id: int) -> Batch:
    bsz = len(batch)
    entity_id = torch.tensor([x["entity_id"] for x in batch], dtype=torch.long)
    input_ids = None
    attention_mask = None
    profile_input_ids = None
    profile_attention_mask = None
    event_input_ids = None
    event_attention_mask = None
    event_times = None
    calendar_features = None
    event_mask = None
    labels = None

    if "input_ids" in batch[0]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            am = torch.tensor(item["attention_mask"], dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(am)] = am

    if "profile_input_ids" in batch[0]:
        profile_input_ids = torch.as_tensor(
            np.stack([_normalize_nested_array(x["profile_input_ids"], np.int64) for x in batch], axis=0),
            dtype=torch.long,
        )
        profile_attention_mask = torch.as_tensor(
            np.stack([_normalize_nested_array(x["profile_attention_mask"], np.bool_) for x in batch], axis=0),
            dtype=torch.bool,
        )
        event_input_ids = torch.as_tensor(
            np.stack([_normalize_nested_array(x["event_input_ids"], np.int64) for x in batch], axis=0),
            dtype=torch.long,
        )
        event_attention_mask = torch.as_tensor(
            np.stack([_normalize_nested_array(x["event_attention_mask"], np.bool_) for x in batch], axis=0),
            dtype=torch.bool,
        )
        event_times = torch.as_tensor(
            np.stack([_normalize_nested_array(x["event_times"], np.float32) for x in batch], axis=0),
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

    if "label" in batch[0]:
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return Batch(
        entity_id=entity_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        profile_input_ids=profile_input_ids,
        profile_attention_mask=profile_attention_mask,
        event_input_ids=event_input_ids,
        event_attention_mask=event_attention_mask,
        event_times=event_times,
        calendar_features=calendar_features,
        event_mask=event_mask,
        labels=labels,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
