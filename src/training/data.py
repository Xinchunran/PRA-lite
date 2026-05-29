from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_ids(path: str | Path) -> set[int]:
    p = Path(path)
    return {int(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip() != ""}


@dataclass(frozen=True)
class Batch:
    entity_id: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None = None


class TokenizedDataset(Dataset):
    def __init__(self, data_path: Path, entity_ids: set[int] | None = None) -> None:
        df = pd.read_parquet(data_path)
        if entity_ids is not None:
            df = df[df["entity_id"].isin(list(entity_ids))]
        self.entity_id = df["entity_id"].astype("int64").to_numpy()
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()
        self.label = df["label"].astype("int64").to_numpy() if "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.entity_id)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "entity_id": int(self.entity_id[idx]),
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.label is not None:
            item["label"] = int(self.label[idx])
        return item


def pad_collate(batch: list[dict], pad_id: int) -> Batch:
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)

    entity_id = torch.tensor([x["entity_id"] for x in batch], dtype=torch.long)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = None

    for i, item in enumerate(batch):
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        am = torch.tensor(item["attention_mask"], dtype=torch.long)
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(am)] = am

    if "label" in batch[0]:
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return Batch(entity_id=entity_id, input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
