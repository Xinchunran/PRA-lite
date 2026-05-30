from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.pragma_lite.model import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint
from src.training.data import load_tokenized_split, pad_collate, set_seed


def _build_loader(ds: TokenizedDataset, pad_id: int, batch_size: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, pad_id=pad_id),
        num_workers=0,
    )


def _select_representation(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    entity_ids: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"collect:{repr_type}"):
            if batch.labels is None:
                raise ValueError("Tokenized dataset must include label for linear_probe")
            model_inputs = {
                key: value.to(device)
                for key, value in batch.model_inputs().items()
                if value is not None
            }
            outputs = model(**model_inputs)
            reps = _select_representation(outputs, model_inputs["event_mask"], repr_type=repr_type)
            features.append(reps.detach().cpu().numpy())
            labels.append(batch.labels.detach().cpu().numpy().astype(np.int64))
            entity_ids.append(batch.entity_id.detach().cpu().numpy())
    return np.concatenate(features), np.concatenate(labels), np.concatenate(entity_ids)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int | None]:
    y_pred = (y_score >= 0.5).astype(np.int64)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "accuracy": float((y_pred == y_true).mean()),
    }


def run_probe_experiment(
    checkpoint: Path,
    data_dir: Path,
    split_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int,
    repr_type: str,
    train_split_name: str = "train",
    eval_splits: tuple[str, ...] = ("valid", "test"),
) -> dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint, map_location=device)
    tokenizer_dir = Path(ckpt["tokenizer_dir"])
    vocab = TokenizerVocab.load(tokenizer_dir)

    cfg = PragmaLiteConfig(**ckpt["model_cfg"])
    model = PragmaLiteModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    train_ds = load_tokenized_split(data_dir, train_split_name, split_dir=split_dir)
    train_loader = _build_loader(train_ds, pad_id=vocab.pad_id, batch_size=batch_size)
    x_train, y_train, train_entity_ids = collect_representations(model, train_loader, device=device, repr_type=repr_type)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(x_train_scaled, y_train)

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "repr_type": repr_type,
        "seed": seed,
        "train": {"n": int(len(y_train)), "pos": int(y_train.sum())},
    }

    train_pred = clf.predict_proba(x_train_scaled)[:, 1]
    report["train_metrics"] = _binary_metrics(y_train, train_pred)

    train_pred_df = pd.DataFrame(
        {
            "entity_id": train_entity_ids.astype(np.int64),
            "label": y_train.astype(np.int64),
            "probability": train_pred.astype(np.float64),
        }
    )
    train_pred_df.to_parquet(output_dir / f"{repr_type}_train_predictions.parquet", index=False)

    for split_name in eval_splits:
        split_ds = load_tokenized_split(data_dir, split_name, split_dir=split_dir)
        split_loader = _build_loader(split_ds, pad_id=vocab.pad_id, batch_size=batch_size)
        x_eval, y_eval, eval_entity_ids = collect_representations(model, split_loader, device=device, repr_type=repr_type)
        y_score = clf.predict_proba(scaler.transform(x_eval))[:, 1]
        report[f"{split_name}_metrics"] = _binary_metrics(y_eval, y_score)
        pd.DataFrame(
            {
                "entity_id": eval_entity_ids.astype(np.int64),
                "label": y_eval.astype(np.int64),
                "probability": y_score.astype(np.float64),
            }
        ).to_parquet(output_dir / f"{repr_type}_{split_name}_predictions.parquet", index=False)

    (output_dir / f"{repr_type}_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{repr_type}_probe.pkl").open("wb") as f:
        pickle.dump({"scaler": scaler, "classifier": clf, "repr_type": repr_type}, f)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--repr_type",
        default="concat",
        choices=["zh_usr", "last_evt", "concat", "record"],
    )
    args = parser.parse_args()

    run_probe_experiment(
        checkpoint=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        split_dir=Path(args.split_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        repr_type=args.repr_type,
    )


if __name__ == "__main__":
    main()
