from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.tokenizer.text_bpe import load_text_tokenizer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[USR]", "[EVT]"]


@dataclass(frozen=True)
class NumericBinner:
    edges: list[float]

    def bucket(self, x: float | int | None) -> int:
        if x is None:
            return -1
        try:
            v = float(x)
        except Exception:
            return -1
        lo = 0
        hi = len(self.edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if v <= self.edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo


class TokenizerVocab:
    def __init__(
        self,
        token_to_id: dict[str, int],
        profile_cols: list[str],
        event_cols: list[str],
        numeric_binners: dict[str, NumericBinner],
        field_value_types: dict[str, str] | None = None,
        categorical_values: dict[str, list[str]] | None = None,
        text_tokenizer_path: str | None = None,
        max_value_tokens_per_field: int = 1,
        tokenizer_version: int = 1,
        numeric_zero_bucket: bool = False,
        text_tokenizer: object | None = None,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.profile_cols = profile_cols
        self.event_cols = event_cols
        self.numeric_binners = numeric_binners
        self.field_value_types = field_value_types or {}
        self.categorical_values = categorical_values or {}
        self.text_tokenizer_path = text_tokenizer_path
        self.max_value_tokens_per_field = int(max_value_tokens_per_field)
        self.tokenizer_version = int(tokenizer_version)
        self.numeric_zero_bucket = bool(numeric_zero_bucket)
        self.text_tokenizer = text_tokenizer

        self.pad_id = token_to_id["[PAD]"]
        self.unk_id = token_to_id["[UNK]"]
        self.mask_id = token_to_id["[MASK]"]
        self.usr_id = token_to_id["[USR]"]
        self.evt_id = token_to_id["[EVT]"]

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def encode_many(self, tokens: Iterable[str]) -> list[int]:
        return [self.encode_token(t) for t in tokens]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "token_to_id": self.token_to_id,
            "profile_cols": self.profile_cols,
            "event_cols": self.event_cols,
            "numeric_binners": {k: {"edges": b.edges} for k, b in self.numeric_binners.items()},
            "field_value_types": self.field_value_types,
            "categorical_values": self.categorical_values,
            "text_tokenizer_path": self.text_tokenizer_path,
            "max_value_tokens_per_field": self.max_value_tokens_per_field,
            "tokenizer_version": self.tokenizer_version,
            "numeric_zero_bucket": self.numeric_zero_bucket,
        }
        (p / "tokenizer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TokenizerVocab":
        p = Path(path)
        payload = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
        binners = {k: NumericBinner(edges=v["edges"]) for k, v in payload.get("numeric_binners", {}).items()}
        text_tokenizer_path = payload.get("text_tokenizer_path")
        resolved_text_tokenizer_path = None
        text_tokenizer = None
        if text_tokenizer_path:
            resolved_text_tokenizer_path = str((p / text_tokenizer_path).resolve()) if not Path(text_tokenizer_path).is_absolute() else text_tokenizer_path
            text_tokenizer = load_text_tokenizer(resolved_text_tokenizer_path)
        return TokenizerVocab(
            token_to_id=payload["token_to_id"],
            profile_cols=payload["profile_cols"],
            event_cols=payload["event_cols"],
            numeric_binners=binners,
            field_value_types=payload.get("field_value_types", {}),
            categorical_values=payload.get("categorical_values", {}),
            text_tokenizer_path=payload.get("text_tokenizer_path"),
            max_value_tokens_per_field=int(payload.get("max_value_tokens_per_field", 1)),
            tokenizer_version=int(payload.get("tokenizer_version", 1)),
            numeric_zero_bucket=bool(payload.get("numeric_zero_bucket", False)),
            text_tokenizer=text_tokenizer,
        )
