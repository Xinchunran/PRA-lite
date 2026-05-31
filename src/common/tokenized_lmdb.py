from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lmdb
    _LMDB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised when lmdb is missing or broken
    lmdb = None
    _LMDB_IMPORT_ERROR = exc


def format_lmdb_key(index: int) -> bytes:
    return f"{index:012d}".encode("ascii")


class TokenizedLmdbWriter:
    def __init__(
        self,
        lmdb_path: Path,
        map_size_gb: int = 64,
        commit_interval: int = 1024,
    ) -> None:
        if lmdb is None:
            detail = f" Original import error: {_LMDB_IMPORT_ERROR!r}" if _LMDB_IMPORT_ERROR is not None else ""
            raise ModuleNotFoundError(
                "lmdb is required for the LMDB backend. Install a working `lmdb` package or use the parquet backend."
                f"{detail}"
            )
        self.path = Path(lmdb_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=max(int(map_size_gb), 1) * 1024**3,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )
        self.commit_interval = max(int(commit_interval), 1)
        self._txn = self.env.begin(write=True)
        self._count = 0
        self._uncommitted = 0
        self._entity_ids: list[int] = []
        self._batching_event_counts: list[int] = []
        self._batching_profile_token_counts: list[int] = []

    def write(self, row: dict[str, Any]) -> None:
        key = format_lmdb_key(self._count)
        self._txn.put(key, pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
        self._entity_ids.append(int(row["entity_id"]))
        self._batching_event_counts.append(int(row.get("batching_event_count", -1)))
        self._batching_profile_token_counts.append(int(row.get("batching_profile_token_count", -1)))
        self._count += 1
        self._uncommitted += 1
        if self._uncommitted >= self.commit_interval:
            self._txn.commit()
            self._txn = self.env.begin(write=True)
            self._uncommitted = 0

    def close(self) -> None:
        self._txn.put(b"__len__", str(self._count).encode("ascii"))
        self._txn.commit()
        np.save(self.path / "entity_ids.npy", np.asarray(self._entity_ids, dtype=np.int64))
        if any(value >= 0 for value in self._batching_event_counts):
            np.save(self.path / "batching_event_counts.npy", np.asarray(self._batching_event_counts, dtype=np.int32))
        if any(value >= 0 for value in self._batching_profile_token_counts):
            np.save(
                self.path / "batching_profile_token_counts.npy",
                np.asarray(self._batching_profile_token_counts, dtype=np.int32),
            )
        (self.path / "length.txt").write_text(f"{self._count}\n", encoding="utf-8")
        self.env.sync()
        self.env.close()
