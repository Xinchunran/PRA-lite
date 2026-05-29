from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint


def _time_delta_bucket(delta_minutes: float | None) -> int:
    if delta_minutes is None or not np.isfinite(delta_minutes) or delta_minutes < 0:
        return 0
    x = float(delta_minutes)
    b = int(np.floor(np.log1p(x)))
    return int(np.clip(b, 0, 31))


def _parse_ts(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _encode_record(vocab: TokenizerVocab, record: dict, max_events: int = 512, max_event_tokens: int = 24) -> tuple[list[int], list[int]]:
    tokens: list[int] = [vocab.usr_id]
    profile = record.get("profile", {}) or {}

    for col in vocab.profile_cols:
        tokens.append(vocab.encode_token(f"KP:{col}"))
        val = profile.get(col, None)
        col_key = f"P:{col}"
        if col_key in vocab.numeric_binners:
            b = vocab.numeric_binners[col_key].bucket(val)
            b = max(b, 0)
            tokens.append(vocab.encode_token(f"VP:{col}#B{b}"))
        else:
            v = "[NA]" if val is None else str(val)
            tokens.append(vocab.encode_token(f"VP:{col}={v}"))

    events = record.get("events", []) or []
    events = events[:max_events]
    last_ts = None
    for ev in events:
        event_tokens: list[int] = [vocab.evt_id]
        ts = _parse_ts(ev.get("timestamp", ""))
        if ts is pd.NaT:
            delta_min = None
        elif last_ts is None or last_ts is pd.NaT:
            delta_min = None
        else:
            delta_min = (ts - last_ts).total_seconds() / 60.0
        last_ts = ts

        dt_bucket = _time_delta_bucket(delta_min)
        event_tokens.append(vocab.encode_token("KE:time_delta"))
        event_tokens.append(vocab.encode_token(f"VE:time_delta#B{dt_bucket}"))

        fields = ev.get("fields", {}) or {}
        for col in vocab.event_cols:
            event_tokens.append(vocab.encode_token(f"KE:{col}"))
            val = fields.get(col, None)
            col_key = f"E:{col}"
            if col_key in vocab.numeric_binners:
                b = vocab.numeric_binners[col_key].bucket(val)
                b = max(b, 0)
                event_tokens.append(vocab.encode_token(f"VE:{col}#B{b}"))
            else:
                v = "[NA]" if val is None else str(val)
                event_tokens.append(vocab.encode_token(f"VE:{col}={v}"))
            if len(event_tokens) >= max_event_tokens:
                break
        tokens.extend(event_tokens[:max_event_tokens])

    attn = [1] * len(tokens)
    return tokens, attn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    vocab = TokenizerVocab.load(Path(ckpt["tokenizer_dir"]))
    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLite(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    record = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    input_ids, attention_mask = _encode_record(vocab, record)
    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=args.device)

    with torch.no_grad():
        h = model(input_ids_t, attention_mask=attention_mask_t)
        logit = float(model.cls_logits(h).detach().cpu().item())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

    out = {"probability": prob, "logit": logit, "timestamp": datetime.utcnow().isoformat() + "Z"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
