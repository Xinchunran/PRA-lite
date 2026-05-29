from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint
from src.training.data import TokenizedDataset, pad_collate, read_ids


def _guess_split_dir(data_dir: Path) -> Path | None:
    parts = list(data_dir.parts)
    if "processed" in parts:
        i = parts.index("processed")
        if i + 1 < len(parts):
            dataset = parts[i + 1]
            return Path(*parts[: i]) / "splits" / dataset
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--split_dir")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint), map_location=args.device)
    tokenizer_dir = Path(ckpt["tokenizer_dir"])
    vocab = TokenizerVocab.load(tokenizer_dir)

    data_dir = Path(args.data_dir)
    ids = None
    if args.split != "all":
        split_dir = Path(args.split_dir) if args.split_dir else _guess_split_dir(data_dir)
        if split_dir is None:
            raise ValueError("Unable to infer split_dir; pass --split_dir explicitly")
        ids = read_ids(split_dir / f"{args.split}_ids.txt")

    ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=ids)
    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model_cls = PragmaLiteModel if ds.has_structured_inputs else PragmaLite
    model = model_cls(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=0,
    )

    out_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="embed"):
            entity_ids = batch.entity_id.detach().cpu().numpy()
            labels = batch.labels.detach().cpu().numpy() if batch.labels is not None else None

            model_inputs = {key: value.to(args.device) for key, value in batch.model_inputs().items() if value is not None}
            h = model(**model_inputs)
            if isinstance(h, dict):
                emb = h["record_embedding"].detach().cpu().numpy()
            else:
                emb = h[:, 0, :].detach().cpu().numpy()

            for i in range(len(entity_ids)):
                row = {"entity_id": int(entity_ids[i])}
                if labels is not None:
                    row["label"] = int(labels[i])
                for j in range(emb.shape[1]):
                    row[f"embedding_{j}"] = float(emb[i, j])
                out_rows.append(row)

    df = pd.DataFrame(out_rows)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
