from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import pandas as pd

from src.model.pragma_lite.model import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.structured import StructuredRecordConfig, encode_record
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    vocab = TokenizerVocab.load(Path(ckpt["tokenizer_dir"]))
    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLiteModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    record = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    evaluation_time = pd.to_datetime(record["evaluation_time"], utc=True, errors="raise")
    encoded = encode_record(
        vocab=vocab,
        profile=record.get("profile", {}) or {},
        events=record.get("events", []) or [],
        evaluation_time=evaluation_time,
        cfg=StructuredRecordConfig(
            max_events=cfg.max_events,
            max_event_tokens=cfg.max_event_tokens,
            max_profile_tokens=cfg.max_profile_tokens,
        ),
    )
    model_inputs = {
        "profile_key_ids": torch.tensor([encoded["profile_key_ids"]], dtype=torch.long, device=args.device),
        "profile_value_ids": torch.tensor([encoded["profile_value_ids"]], dtype=torch.long, device=args.device),
        "profile_value_pos": torch.tensor([encoded["profile_value_pos"]], dtype=torch.long, device=args.device),
        "profile_time": torch.tensor([encoded["profile_time"]], dtype=torch.float32, device=args.device),
        "profile_mask": torch.tensor([encoded["profile_mask"]], dtype=torch.bool, device=args.device),
        "event_key_ids": torch.tensor([encoded["event_key_ids"]], dtype=torch.long, device=args.device),
        "event_value_ids": torch.tensor([encoded["event_value_ids"]], dtype=torch.long, device=args.device),
        "event_value_pos": torch.tensor([encoded["event_value_pos"]], dtype=torch.long, device=args.device),
        "event_token_mask": torch.tensor([encoded["event_token_mask"]], dtype=torch.bool, device=args.device),
        "event_time": torch.tensor([encoded["event_time"]], dtype=torch.float32, device=args.device),
        "calendar_features": torch.tensor([encoded["calendar_features"]], dtype=torch.float32, device=args.device),
        "event_mask": torch.tensor([encoded["event_mask"]], dtype=torch.bool, device=args.device),
    }

    with torch.no_grad():
        h = model(**model_inputs)
        logit = float(model.cls_logits(h).detach().cpu().item())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

    out = {"probability": prob, "logit": logit, "timestamp": datetime.utcnow().isoformat() + "Z"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
