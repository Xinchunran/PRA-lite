from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_downloader.build_events import _drop_forbidden_label_columns
from src.common.yaml_utils import load_yaml


def test_transxion_config_does_not_include_label_column_in_transaction_inputs() -> None:
    cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "data" / "transxion.yaml")
    label_col = str(cfg["label_col"])
    transaction_columns = [str(col) for col in cfg.get("transaction_columns", [])]
    assert label_col not in transaction_columns, (
        f"Label column {label_col!r} must not appear in transaction_columns: {transaction_columns}"
    )


def test_build_events_hard_drops_target_like_columns_from_event_inputs() -> None:
    events = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "amount": [10.0, 20.0],
            "is_laundering": [0, 1],
            "label": [0, 1],
            "target": [0, 1],
            "is_fraud": [0, 0],
        }
    )

    cleaned = _drop_forbidden_label_columns(events, label_col="is_laundering")

    assert "amount" in cleaned.columns
    assert "is_laundering" not in cleaned.columns
    assert "label" not in cleaned.columns
    assert "target" not in cleaned.columns
    assert "is_fraud" not in cleaned.columns
