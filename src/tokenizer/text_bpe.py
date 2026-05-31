from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable


_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TextEncoding:
    tokens: list[str]


class SimpleTextTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = list(vocab)
        self.vocab_set = set(vocab)

    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text).lower().strip()

    def encode(self, text: str) -> TextEncoding:
        normalized = self.normalize(text)
        if not normalized:
            return TextEncoding(tokens=[])
        tokens = [piece for piece in _WORD_RE.findall(normalized) if piece]
        if not tokens:
            return TextEncoding(tokens=[])
        return TextEncoding(tokens=[piece if piece in self.vocab_set else "[UNK]" for piece in tokens])

    def save(self, path: str | Path) -> None:
        payload = {"type": "simple_whitespace", "vocab": self.vocab}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTextTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=list(payload.get("vocab", ["[UNK]"])))


def train_text_tokenizer(
    texts: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int,
) -> tuple[object, str]:
    output_path = Path(output_path)
    texts = list(texts)
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import Lowercase, NFKC, Sequence
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.save(str(output_path))
        return tokenizer, "hf_bpe"
    except Exception:
        counter: Counter[str] = Counter()
        for text in texts:
            normalized = SimpleTextTokenizer.normalize(text)
            counter.update(piece for piece in _WORD_RE.findall(normalized) if piece)
        vocab = ["[UNK]"] + [token for token, _ in counter.most_common(max(vocab_size - 1, 0))]
        tokenizer = SimpleTextTokenizer(vocab=vocab)
        tokenizer.save(output_path)
        return tokenizer, "simple_whitespace"


def load_text_tokenizer(path: str | Path) -> object:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text tokenizer file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload.get("type") == "simple_whitespace":
        return SimpleTextTokenizer.load(path)
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))
    except Exception:
        return SimpleTextTokenizer.load(path)
