from __future__ import annotations

from collections.abc import Iterable, Mapping


def validate_splits(splits: Mapping[str, Iterable[str]]) -> None:
    split_sets = {name: set(values) for name, values in splits.items()}
    names = list(split_sets.keys())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_sets[left].intersection(split_sets[right])
            if overlap:
                raise ValueError(f"Entity leakage between {left} and {right}: {sorted(overlap)[:5]}")
