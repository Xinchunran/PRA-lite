from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.yaml_utils import load_yaml
from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import TokenizedDataset, pad_collate, read_ids, set_seed


def _load_model_from_checkpoint(ckpt_path: Path, device: str) -> tuple[PragmaLite, TokenizerVocab, dict]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    tokenizer_dir = Path(ckpt["tokenizer_dir"])
    vocab = TokenizerVocab.load(tokenizer_dir)
    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLite(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, vocab, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    seed = int(train_cfg["training"].get("seed", 42))
    set_seed(seed)

    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, ckpt = _load_model_from_checkpoint(Path(args.checkpoint), args.device)

    train_ids = read_ids(split_dir / "train_ids.txt")
    valid_ids = read_ids(split_dir / "valid_ids.txt")
    train_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=train_ids)
    valid_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=valid_ids)

    batch_size = int(train_cfg["training"].get("batch_size", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=0,
    )

    lr = float(train_cfg["training"].get("learning_rate", 1e-4))
    wd = float(train_cfg["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    max_epochs = int(train_cfg["training"].get("max_epochs", 3))
    log_every = int(train_cfg["training"].get("log_every", 50))

    best_metric = -float("inf")

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        n = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"finetune epoch {epoch+1}/{max_epochs}")):
            input_ids = batch.input_ids.to(args.device)
            attention_mask = batch.attention_mask.to(args.device)
            labels = batch.labels.to(args.device) if batch.labels is not None else None
            if labels is None:
                raise ValueError("Tokenized dataset must include label for finetune")

            h = model(input_ids, attention_mask=attention_mask)
            logits = model.cls_logits(h)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * len(input_ids)
            n += len(input_ids)

            if (step + 1) % log_every == 0:
                pass

        train_loss = running / max(n, 1)

        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for batch in valid_loader:
                input_ids = batch.input_ids.to(args.device)
                attention_mask = batch.attention_mask.to(args.device)
                labels = batch.labels.to(args.device) if batch.labels is not None else None
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
                    "task": "binary_classification",
                    "model_state": model.state_dict(),
                    "model_cfg": model.cfg.__dict__,
                    "tokenizer_dir": ckpt["tokenizer_dir"],
                    "best_valid_pr_auc": best_metric,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                },
            )


if __name__ == "__main__":
    main()
