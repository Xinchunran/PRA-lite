from __future__ import annotations

import numpy as np


class PercentileBucketizer:
    def __init__(self, num_buckets: int = 10, add_zero_bucket: bool = True) -> None:
        if num_buckets < 2:
            raise ValueError("num_buckets must be at least 2")
        self.num_buckets = num_buckets
        self.add_zero_bucket = add_zero_bucket
        self.boundaries_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "PercentileBucketizer":
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if self.add_zero_bucket:
            arr = arr[arr != 0.0]

        if arr.size == 0:
            self.boundaries_ = np.array([], dtype=np.float64)
            return self

        effective_buckets = self.num_buckets - 1 if self.add_zero_bucket else self.num_buckets
        quantiles = np.linspace(0.0, 1.0, effective_buckets + 1)[1:-1]
        self.boundaries_ = np.quantile(arr, quantiles).astype(np.float64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.boundaries_ is None:
            raise ValueError("Bucketizer must be fit before transform")

        arr = np.asarray(values, dtype=np.float64)
        out = np.zeros(arr.shape, dtype=np.int64)

        nonzero_mask = np.isfinite(arr)
        if self.add_zero_bucket:
            nonzero_mask &= arr != 0.0

        bucket_ids = np.searchsorted(self.boundaries_, arr[nonzero_mask], side="left")
        if self.add_zero_bucket:
            bucket_ids = bucket_ids + 1
        out[nonzero_mask] = bucket_ids
        return out
