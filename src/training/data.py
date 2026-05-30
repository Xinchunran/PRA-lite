from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.common.fs import read_json
from src.common.tokenized_lmdb import format_lmdb_key

try:
    import lmdb
    _LMDB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised when lmdb backend is unavailable or broken
    lmdb = None
    _LMDB_IMPORT_ERROR = exc

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


def _debug_event(event: str, hypothesis_id: str, location: str, **payload: object) -> None:
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
        "hypothesisId": hypothesis_id,
        "location": location,
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


def load_tokenized_manifest_split(manifest_path: str | Path, split_name: str) -> Dataset:
    manifest = read_json(manifest_path)
    shard_entries = manifest.get("shards", [])
    datasets: list[Dataset] = []
    for entry in shard_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "ready")) != "ready":
            continue
        tokenized_dir_raw = entry.get("tokenized_dir")
        if not tokenized_dir_raw:
            continue
        tokenized_dir = Path(str(tokenized_dir_raw))
        split_lmdb_path = tokenized_dir / f"{split_name}.lmdb"
        if split_lmdb_path.exists():
            datasets.append(LmdbTokenizedDataset(split_lmdb_path))
            continue
        split_parquet_path = tokenized_dir / f"{split_name}.parquet"
        if split_parquet_path.exists():
            datasets.append(TokenizedDataset(split_parquet_path))
    if not datasets:
        raise FileNotFoundError(f"Manifest {manifest_path} has no ready shard for split {split_name}")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


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
        # #region debug-point A:tokenized-dataset-init
        init_started_at = time.perf_counter()
        # #endregion
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
        self._debug_logged_row_groups: set[int] = set()
        # #region debug-point A:tokenized-dataset-init
        _debug_event(
            "tokenized_dataset_init",
            "A",
            "src/training/data.py:127",
            path=str(self.data_path),
            row_groups=self._parquet.num_row_groups,
            num_rows=self._num_rows,
            filtered=entity_ids is not None,
            elapsed_s=round(time.perf_counter() - init_started_at, 4),
        )
        # #endregion

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
        # #region debug-point C:row-group-read
        load_started_at = time.perf_counter()
        # #endregion
        table = self._parquet.read_row_group(row_group_idx, columns=self._columns)
        row_group_data = {column: values for column, values in table.to_pydict().items()}
        self._row_group_cache[row_group_idx] = row_group_data
        if len(self._row_group_cache) > self._max_cached_row_groups:
            self._row_group_cache.popitem(last=False)
        # #region debug-point C:row-group-read
        if len(self._debug_logged_row_groups) < 5 and row_group_idx not in self._debug_logged_row_groups:
            self._debug_logged_row_groups.add(row_group_idx)
            _debug_event(
                "parquet_row_group_loaded",
                "C",
                "src/training/data.py:188",
                path=str(self.data_path),
                row_group_idx=row_group_idx,
                rows=len(next(iter(row_group_data.values()))) if row_group_data else 0,
                cache_size=len(self._row_group_cache),
                elapsed_s=round(time.perf_counter() - load_started_at, 4),
            )
        # #endregion
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
        # #region debug-point A:lmdb-dataset-init
        init_started_at = time.perf_counter()
        # #endregion
        if lmdb is None:
            detail = f" Original import error: {_LMDB_IMPORT_ERROR!r}" if _LMDB_IMPORT_ERROR is not None else ""
            raise ModuleNotFoundError(
                "lmdb is required for the LMDB backend. Install a working `lmdb` package or use the parquet backend."
                f"{detail}"
            )
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
        self._debug_get_count = 0
        # #region debug-point A:lmdb-dataset-init
        _debug_event(
            "lmdb_dataset_init",
            "A",
            "src/training/data.py:224",
            path=str(self.lmdb_path),
            num_rows=int(len(self._indices)),
            filtered=entity_ids is not None,
            elapsed_s=round(time.perf_counter() - init_started_at, 4),
        )
        # #endregion

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
        # #region debug-point C:lmdb-read
        read_started_at = time.perf_counter()
        # #endregion
        real_idx = int(self._indices[idx])
        with self._get_env().begin(write=False) as txn:
            payload = txn.get(format_lmdb_key(real_idx))
        if payload is None:
            raise KeyError(real_idx)
        row = pickle.loads(payload)
        # #region debug-point C:lmdb-read
        self._debug_get_count += 1
        if self._debug_get_count <= 5:
            _debug_event(
                "lmdb_item_loaded",
                "C",
                "src/training/data.py:267",
                path=str(self.lmdb_path),
                index=real_idx,
                elapsed_s=round(time.perf_counter() - read_started_at, 4),
            )
        # #endregion
        return row


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
