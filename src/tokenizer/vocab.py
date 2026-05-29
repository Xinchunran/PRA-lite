from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.profile_cols = profile_cols
        self.event_cols = event_cols
        self.numeric_binners = numeric_binners

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
        }
        (p / "tokenizer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TokenizerVocab":
        p = Path(path)
        payload = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
        binners = {k: NumericBinner(edges=v["edges"]) for k, v in payload.get("numeric_binners", {}).items()}
        return TokenizerVocab(
            token_to_id=payload["token_to_id"],
            profile_cols=payload["profile_cols"],
            event_cols=payload["event_cols"],
            numeric_binners=binners,
        )
