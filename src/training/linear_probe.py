from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import TokenizedDataset, pad_collate, read_ids, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    tokenizer_dir = Path(ckpt["tokenizer_dir"])
    vocab = TokenizerVocab.load(tokenizer_dir)

    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLite(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.cls_head.parameters():
        p.requires_grad = True

    train_ids = read_ids(split_dir / "train_ids.txt")
    valid_ids = read_ids(split_dir / "valid_ids.txt")
    train_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=train_ids)
    valid_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=valid_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.cls_head.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -float("inf")
    for epoch in range(args.max_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"probe epoch {epoch+1}/{args.max_epochs}"):
            input_ids = batch.input_ids.to(args.device)
            attention_mask = batch.attention_mask.to(args.device)
            labels = batch.labels.to(args.device) if batch.labels is not None else None
            if labels is None:
                raise ValueError("Tokenized dataset must include label for linear_probe")

            with torch.no_grad():
                h = model(input_ids, attention_mask=attention_mask)
            logits = model.cls_logits(h)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for batch in valid_loader:
                input_ids = batch.input_ids.to(args.device)
                attention_mask = batch.attention_mask.to(args.device)
                labels = batch.labels.to(args.device)
                h = model(input_ids, attention_mask=attention_mask)
                logits = model.cls_logits(h)
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
            y_score = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits))) if all_logits else np.array([])
            y_true = np.concatenate(all_labels).astype(np.int64) if all_labels else np.array([])
            pr_auc = float(average_precision_score(y_true, y_score)) if len(y_true) else 0.0

        if pr_auc > best_metric:
            best_metric = pr_auc
            save_checkpoint(
                out_dir / "best.ckpt",
                {
                    "task": "linear_probe",
                    "model_state": model.state_dict(),
                    "model_cfg": model.cfg.__dict__,
                    "tokenizer_dir": str(tokenizer_dir),
                    "best_valid_pr_auc": best_metric,
                    "epoch": epoch,
                },
            )


if __name__ == "__main__":
    main()
