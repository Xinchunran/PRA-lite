from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.pragma_lite.model import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint
from src.training.data import load_tokenized_dataset, load_tokenized_split, pad_collate


REPRESENTATION_TYPES = ("zh_usr", "last_evt", "concat", "record")


@dataclass(frozen=True)
class ExtractorArtifacts:
    model: PragmaLiteModel
    vocab: TokenizerVocab
    checkpoint: dict[str, object]


def guess_split_dir(data_dir: Path) -> Path | None:
    parts = list(data_dir.parts)
    if "processed" in parts:
        i = parts.index("processed")
        if i + 1 < len(parts):
            dataset = parts[i + 1]
            return Path(*parts[: i]) / "splits" / dataset
    return None


def load_extractor_artifacts(checkpoint_path: Path, device: str) -> ExtractorArtifacts:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    tokenizer_dir = Path(str(ckpt["tokenizer_dir"]))
    vocab = TokenizerVocab.load(tokenizer_dir)
    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLiteModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return ExtractorArtifacts(model=model, vocab=vocab, checkpoint=ckpt)


def build_tokenized_loader(ds: Dataset, pad_id: int, batch_size: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_id=pad_id),
        num_workers=0,
    )


def select_representation(
    outputs: dict[str, torch.Tensor],
    batch_event_mask: torch.Tensor,
    repr_type: str,
) -> torch.Tensor:
    zh_usr = outputs["zh_usr"]
    zh_evt = outputs["zh_evt"]
    if zh_evt.size(1) == 0:
        last_evt = torch.zeros_like(zh_usr)
    else:
        last_idx = batch_event_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, zh_evt.size(-1))
        last_evt = zh_evt.gather(1, gather_idx).squeeze(1)
    if repr_type == "zh_usr":
        return zh_usr
    if repr_type == "last_evt":
        return last_evt
    if repr_type == "concat":
        return torch.cat([zh_usr, last_evt], dim=-1)
    if repr_type == "record":
        return outputs["record_embedding"]
    raise ValueError(f"Unknown repr_type: {repr_type}")


def collect_representations(
    model: PragmaLiteModel,
    loader: DataLoader,
    device: str,
    repr_type: str,
    require_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    entity_ids: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract:{repr_type}"):
            if require_labels and batch.labels is None:
                raise ValueError("Tokenized dataset must include label for representation extraction")
            model_inputs = {
                key: value.to(device)
                for key, value in batch.model_inputs().items()
                if value is not None
            }
            outputs = model(**model_inputs)
            reps = select_representation(outputs, model_inputs["event_mask"], repr_type=repr_type)
            features.append(reps.detach().cpu().numpy())
            entity_ids.append(batch.entity_id.detach().cpu().numpy().astype(np.int64))
            if batch.labels is not None:
                labels.append(batch.labels.detach().cpu().numpy().astype(np.int64))
    if not features:
        return np.empty((0, 0), dtype=np.float32), None, np.empty((0,), dtype=np.int64)
    feature_array = np.concatenate(features, axis=0)
    entity_array = np.concatenate(entity_ids, axis=0)
    label_array = np.concatenate(labels, axis=0) if labels else None
    return feature_array, label_array, entity_array


def extract_embeddings_dataframe(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    split_dir: Path | None,
    batch_size: int,
    device: str,
    repr_type: str,
) -> pd.DataFrame:
    artifacts = load_extractor_artifacts(checkpoint, device=device)
    if split != "all":
        inferred_split_dir = split_dir or guess_split_dir(data_dir)
        if inferred_split_dir is None:
            raise ValueError("Unable to infer split_dir; pass --split_dir explicitly")
        ds = load_tokenized_split(data_dir, split, split_dir=inferred_split_dir)
    else:
        ds = load_tokenized_dataset(data_dir)
    loader = build_tokenized_loader(ds, pad_id=artifacts.vocab.pad_id, batch_size=batch_size)
    features, labels, entity_ids = collect_representations(
        artifacts.model,
        loader,
        device=device,
        repr_type=repr_type,
    )
    rows: dict[str, np.ndarray] = {"entity_id": entity_ids.astype(np.int64)}
    if labels is not None:
        rows["label"] = labels.astype(np.int64)
    for i in range(features.shape[1]):
        rows[f"embedding_{i}"] = features[:, i].astype(np.float32)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--split_dir")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repr_type", default="record", choices=REPRESENTATION_TYPES)
    args = parser.parse_args()

    df = extract_embeddings_dataframe(
        checkpoint=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        split=args.split,
        split_dir=Path(args.split_dir) if args.split_dir else None,
        batch_size=args.batch_size,
        device=args.device,
        repr_type=args.repr_type,
    )
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
