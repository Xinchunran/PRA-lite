from __future__ import annotations

from src.tokenizer.vocab import NumericBinner, TokenizerVocab


def test_vocab_v2_roundtrip_preserves_extended_metadata(tmp_path) -> None:
    vocab = TokenizerVocab(
        token_to_id={
            "[PAD]": 0,
            "[UNK]": 1,
            "[MASK]": 2,
            "[USR]": 3,
            "[EVT]": 4,
            "K:E:description": 5,
            "T:[UNK]": 6,
            "T:metal": 7,
            "V:E:currency=[UNK]": 8,
            "V:E:currency=usd": 9,
            "V:E:amount#[NA]": 10,
            "V:E:amount#ZERO": 11,
            "V:E:amount#B0": 12,
        },
        profile_cols=[],
        event_cols=["description", "currency", "amount"],
        numeric_binners={"E:amount": NumericBinner(edges=[10.0])},
        field_value_types={"E:description": "textual", "E:currency": "categorical", "E:amount": "numeric"},
        categorical_values={"E:currency": ["usd"]},
        text_tokenizer_path="text_bpe.json",
        max_value_tokens_per_field=4,
        tokenizer_version=2,
        numeric_zero_bucket=True,
    )
    (tmp_path / "text_bpe.json").write_text('{"type":"simple_whitespace","vocab":["[UNK]","metal","plan"]}\n', encoding="utf-8")

    vocab.save(tmp_path)
    loaded = TokenizerVocab.load(tmp_path)

    assert loaded.tokenizer_version == 2
    assert loaded.max_value_tokens_per_field == 4
    assert loaded.numeric_zero_bucket is True
    assert loaded.field_value_types["E:description"] == "textual"
    assert loaded.categorical_values["E:currency"] == ["usd"]
    assert loaded.text_tokenizer is not None
