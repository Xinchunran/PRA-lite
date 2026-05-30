from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from src.common.tokenized_lmdb import format_lmdb_key

try:
    import lmdb
except ModuleNotFoundError:  # pragma: no cover - exercised when lmdb backend is unavailable
    lmdb = None

REQUIRED_TOKENIZED_COLUMNS = {
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


def _normalize_nested_array(value: object, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=dtype)


def read_ids(path: str | Path) -> set[int]:
    p = Path(path)
    return {int(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip() != ""}


def _as_python(value: object) -> object:
    return value.as_py() if hasattr(value, "as_py") else value


def load_tokenized_dataset(
    data_dir: str | Path,
    split_name: str | None = None,
    split_dir: str | Path | None = None,
) -> "Dataset":
    data_path = Path(data_dir)
    if split_name is not None:
        split_lmdb_path = data_path / f"{split_name}.lmdb"
        if split_lmdb_path.exists():
            return LmdbTokenizedDataset(split_lmdb_path)
        split_path = data_path / f"{split_name}.parquet"
        if split_path.exists():
            return TokenizedDataset(split_path)
    full_lmdb_path = data_path / "dataset.lmdb"
    if full_lmdb_path.exists():
        entity_ids = None
        if split_name is not None:
            if split_dir is None:
                raise FileNotFoundError(f"Missing split metadata for LMDB fallback: {split_name}")
            entity_ids = read_ids(Path(split_dir) / f"{split_name}_ids.txt")
        return LmdbTokenizedDataset(full_lmdb_path, entity_ids=entity_ids)
    full_parquet_path = data_path / "dataset.parquet"
    if full_parquet_path.exists():
        entity_ids = None
        if split_name is not None:
            if split_dir is None:
                raise FileNotFoundError(f"Missing split metadata for parquet fallback: {split_name}")
            entity_ids = read_ids(Path(split_dir) / f"{split_name}_ids.txt")
        return TokenizedDataset(full_parquet_path, entity_ids=entity_ids)
    if split_name is not None:
        raise FileNotFoundError(f"Missing tokenized dataset for split {split_name} under {data_path}")
    raise FileNotFoundError(f"Missing tokenized dataset under {data_path}")


def load_tokenized_split(data_dir: str | Path, split_name: str, split_dir: str | Path | None = None) -> "Dataset":
    return load_tokenized_dataset(data_dir, split_name=split_name, split_dir=split_dir)


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
    def __init__(self, data_path: Path, entity_ids: set[int] | None = None, max_cached_row_groups: int = 2) -> None:
        self.data_path = Path(data_path)
        self._parquet = pq.ParquetFile(self.data_path)
        schema_names = set(self._parquet.schema_arrow.names)
        missing = REQUIRED_TOKENIZED_COLUMNS.difference(schema_names)
        if missing:
            raise ValueError(
                "Tokenized dataset must contain structured PRAGMA columns only; missing: "
                + ", ".join(sorted(missing))
            )
        self._columns = [
            "entity_id",
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
        ]
        self._has_label = "label" in schema_names
        if self._has_label:
            self._columns.append("label")
        self._row_group_starts: list[int] = []
        total_rows = 0
        for row_group_idx in range(self._parquet.num_row_groups):
            self._row_group_starts.append(total_rows)
            total_rows += self._parquet.metadata.row_group(row_group_idx).num_rows
        self._num_rows = int(total_rows)
        self._entity_row_index: list[tuple[int, int]] | None = None
        if entity_ids is not None:
            self._entity_row_index = []
            for row_group_idx in range(self._parquet.num_row_groups):
                entity_table = self._parquet.read_row_group(row_group_idx, columns=["entity_id"])
                entity_values = entity_table.column("entity_id").to_pylist()
                for row_idx, entity_id in enumerate(entity_values):
                    if int(entity_id) in entity_ids:
                        self._entity_row_index.append((row_group_idx, row_idx))
            self._num_rows = len(self._entity_row_index)
        self._max_cached_row_groups = max(1, int(max_cached_row_groups))
        self._row_group_cache: OrderedDict[int, dict[str, list[object]]] = OrderedDict()

    def __len__(self) -> int:
        return self._num_rows

    def _resolve_row_position(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._entity_row_index is not None:
            return self._entity_row_index[idx]
        row_group_idx = bisect_right(self._row_group_starts, idx) - 1
        row_offset = idx - self._row_group_starts[row_group_idx]
        return row_group_idx, row_offset

    def _load_row_group(self, row_group_idx: int) -> dict[str, list[object]]:
        cached = self._row_group_cache.get(row_group_idx)
        if cached is not None:
            self._row_group_cache.move_to_end(row_group_idx)
            return cached
        table = self._parquet.read_row_group(row_group_idx, columns=self._columns)
        row_group_data = {column: values for column, values in table.to_pydict().items()}
        self._row_group_cache[row_group_idx] = row_group_data
        if len(self._row_group_cache) > self._max_cached_row_groups:
            self._row_group_cache.popitem(last=False)
        return row_group_data

    def __getitem__(self, idx: int) -> dict:
        row_group_idx, row_idx = self._resolve_row_position(idx)
        row_group = self._load_row_group(row_group_idx)
        item = {
            "entity_id": int(_as_python(row_group["entity_id"][row_idx])),
            "profile_key_ids": _as_python(row_group["profile_key_ids"][row_idx]),
            "profile_value_ids": _as_python(row_group["profile_value_ids"][row_idx]),
            "profile_value_pos": _as_python(row_group["profile_value_pos"][row_idx]),
            "profile_time": _as_python(row_group["profile_time"][row_idx]),
            "profile_mask": _as_python(row_group["profile_mask"][row_idx]),
            "event_key_ids": _as_python(row_group["event_key_ids"][row_idx]),
            "event_value_ids": _as_python(row_group["event_value_ids"][row_idx]),
            "event_value_pos": _as_python(row_group["event_value_pos"][row_idx]),
            "event_token_mask": _as_python(row_group["event_token_mask"][row_idx]),
            "event_time": _as_python(row_group["event_time"][row_idx]),
            "calendar_features": _as_python(row_group["calendar_features"][row_idx]),
            "event_mask": _as_python(row_group["event_mask"][row_idx]),
        }
        if self._has_label:
            item["label"] = int(_as_python(row_group["label"][row_idx]))
        return item


class LmdbTokenizedDataset(Dataset):
    def __init__(self, lmdb_path: Path, entity_ids: set[int] | None = None) -> None:
        if lmdb is None:
            raise ModuleNotFoundError("lmdb is required for the LMDB backend. Install it with `pip install lmdb`.")
        self.lmdb_path = Path(lmdb_path)
        self._env: Any = None
        entity_id_path = self.lmdb_path / "entity_ids.npy"
        if not entity_id_path.exists():
            raise FileNotFoundError(f"Missing LMDB index file: {entity_id_path}")
        self._all_entity_ids = np.load(entity_id_path)
        if entity_ids is None:
            self._indices = np.arange(len(self._all_entity_ids), dtype=np.int64)
        else:
            allowed = set(int(x) for x in entity_ids)
            self._indices = np.asarray(
                [idx for idx, entity_id in enumerate(self._all_entity_ids.tolist()) if int(entity_id) in allowed],
                dtype=np.int64,
            )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __len__(self) -> int:
        return int(len(self._indices))

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
                max_readers=2048,
            )
        return self._env

    def __getitem__(self, idx: int) -> dict:
        real_idx = int(self._indices[idx])
        with self._get_env().begin(write=False) as txn:
            payload = txn.get(format_lmdb_key(real_idx))
        if payload is None:
            raise KeyError(real_idx)
        return pickle.loads(payload)


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
