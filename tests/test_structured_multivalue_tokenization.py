from __future__ import annotations

import pandas as pd

from src.tokenizer.structured import encode_event_features
from src.tokenizer.vocab import TokenizerVocab


def _make_vocab() -> TokenizerVocab:
    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[USR]": 3,
        "[EVT]": 4,
        "K:E:description": 5,
        "K:E:direction": 6,
        "K:E:amount": 7,
        "T:[UNK]": 8,
        "T:[NA]": 9,
        "T:metal": 10,
        "T:plan": 11,
        "V:E:direction=[UNK]": 12,
        "V:E:direction=[NA]": 13,
        "V:E:direction=out": 14,
        "V:E:amount#[NA]": 15,
        "V:E:amount#ZERO": 16,
        "V:E:amount#B0": 17,
        "V:E:amount#B1": 18,
    }
    return TokenizerVocab(
        token_to_id=token_to_id,
        profile_cols=[],
        event_cols=["description", "direction", "amount"],
        numeric_binners={"E:amount": type("Binner", (), {"bucket": lambda self, x: 1 if float(x) > 0 else 0})()},
        field_value_types={"E:description": "textual", "E:direction": "categorical", "E:amount": "numeric"},
        categorical_values={"E:direction": ["out"]},
        max_value_tokens_per_field=4,
        tokenizer_version=2,
        numeric_zero_bucket=True,
        text_tokenizer=type("Tokenizer", (), {"encode": lambda self, text: type("Enc", (), {"tokens": ["metal", "plan"]})()})(),
    )


def test_text_field_replicates_key_and_positions() -> None:
    vocab = _make_vocab()
    encoded = encode_event_features(
        vocab=vocab,
        events=[
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "description": "metal plan",
                "direction": "out",
                "amount": 12.0,
            }
        ],
        evaluation_time=pd.Timestamp("2024-01-02T00:00:00Z"),
        max_events=1,
        max_event_tokens=6,
    )

    assert encoded["event_key_ids"][0][:4] == [5, 5, 6, 7]
    assert encoded["event_value_ids"][0][:4] == [10, 11, 14, 18]
    assert encoded["event_value_pos"][0][:4] == [0, 1, 0, 0]


def test_multivalue_padding_mask_stays_consistent() -> None:
    vocab = _make_vocab()
    encoded = encode_event_features(
        vocab=vocab,
        events=[
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "description": "metal plan",
                "direction": "out",
                "amount": 0.0,
            }
        ],
        evaluation_time=pd.Timestamp("2024-01-02T00:00:00Z"),
        max_events=1,
        max_event_tokens=6,
    )

    assert sum(encoded["event_token_mask"][0]) == 4
    assert encoded["event_value_ids"][0][3] == 16
    assert all(
        pos == 0
        for pos, mask in zip(encoded["event_value_pos"][0], encoded["event_token_mask"][0])
        if mask == 0
    )
