from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSchema:
    namespace: str
    name: str
    value_type: str
    cardinality: int | None = None


@dataclass(frozen=True)
class VocabBuildConfig:
    num_numeric_bins: int = 100
    categorical_threshold: int = 2048
    max_text_vocab_size: int = 28000
    max_value_tokens_per_field: int = 4
    numeric_zero_bucket: bool = True
    force_categorical_cols: tuple[str, ...] = ("currency", "mcc", "type", "direction")
    force_textual_cols: tuple[str, ...] = ("description", "merchant_name", "memo")
