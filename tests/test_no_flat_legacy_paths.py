from __future__ import annotations

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def test_no_flat_legacy_symbols_remain_in_source() -> None:
    banned_tokens = [
        "class PragmaLite(",
        "_split_flat_inputs",
        "profile_input_ids",
        "profile_attention_mask",
        "event_input_ids",
        "event_attention_mask",
        "input_ids",
        "attention_mask",
        "mlm_probability",
    ]

    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned_tokens:
            if token in text:
                violations.append(f"{path.relative_to(SRC_ROOT)} -> {token}")

    assert not violations, "Legacy flat-path symbols remain:\n" + "\n".join(sorted(violations))
