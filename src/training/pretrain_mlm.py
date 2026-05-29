from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.common.yaml_utils import load_yaml
from src.model.pragma_lite.model import PragmaLite, PragmaLiteConfig
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import save_checkpoint
from src.training.data import TokenizedDataset, pad_collate, read_ids, set_seed


def _guess_tokenizer_dir(data_dir: Path) -> Path:
    candidate = data_dir.parent / "tokenizer"
    return candidate


def _make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_id: int,
    pad_id: int,
    special_ids: set[int],
    mask_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = input_ids.shape
    labels = torch.full((bsz, seq_len), -100, dtype=torch.long, device=input_ids.device)

    eligible = attention_mask.bool()
    for sid in special_ids:
        eligible = eligible & (input_ids != sid)
    eligible = eligible & (input_ids != pad_id)

    mask = (torch.rand((bsz, seq_len), device=input_ids.device) < mask_prob) & eligible
    labels[mask] = input_ids[mask]

    masked_input = input_ids.clone()
    masked_input[mask] = mask_id
    return masked_input, labels


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _unwrap_model(model: nn.Module) -> PragmaLite:
    return model.module if isinstance(model, DDP) else model


def _setup_device(requested_device: str, backend: str) -> tuple[torch.device, int | None]:
    if _is_distributed():
        if not torch.cuda.is_available():
            raise ValueError("DDP training requires CUDA GPUs and torchrun.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device(requested_device), None


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    vocab: TokenizerVocab,
    mask_prob: float,
    device: torch.device,
) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            masked_input, mlm_labels = _make_mlm_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_id=vocab.mask_id,
                pad_id=vocab.pad_id,
                special_ids={vocab.usr_id, vocab.evt_id},
                mask_prob=mask_prob,
            )
            logits = model(masked_input, attention_mask=attention_mask, return_mlm_logits=True)
            loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)["model"]
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg["training"].get("seed", 42))
    set_seed(seed)
    device, local_rank = _setup_device(args.device, args.backend)
    is_main = _is_main_process()

    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else _guess_tokenizer_dir(data_dir)
    vocab = TokenizerVocab.load(tokenizer_dir)

    cfg = PragmaLiteConfig(
        vocab_size=len(vocab.token_to_id),
        d_model=int(model_cfg.get("d_model", 192)),
        n_heads=int(model_cfg.get("n_heads", 3)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=int(model_cfg.get("max_seq_len", 4096)),
    )
    model: nn.Module = PragmaLite(cfg).to(device)
    if _is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    train_ids = read_ids(split_dir / "train_ids.txt")
    valid_ids = read_ids(split_dir / "valid_ids.txt")
    train_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=train_ids)
    valid_ds = TokenizedDataset(data_dir / "dataset.parquet", entity_ids=valid_ids)

    batch_size = int(train_cfg["training"].get("batch_size", 32))
    num_workers = int(train_cfg["training"].get("num_workers", 0))
    pin_memory = bool(train_cfg["training"].get("pin_memory", torch.cuda.is_available()))
    train_sampler = None
    if _is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    valid_loader = None
    if not _is_distributed() or is_main:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: pad_collate(b, pad_id=vocab.pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    lr = float(train_cfg["training"].get("learning_rate", 3e-4))
    wd = float(train_cfg["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    max_steps = int(train_cfg["training"].get("max_steps", 1000))
    log_every = int(train_cfg["training"].get("log_every", 50))
    mask_prob = float(train_cfg.get("masking", {}).get("token_mask_prob", 0.15))

    best_valid = float("inf")
    step = 0
    model.train()
    epoch = 0
    pbar = tqdm(total=max_steps, desc="pretrain", disable=not is_main)
    try:
        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= max_steps:
                    break
                input_ids = batch.input_ids.to(device, non_blocking=True)
                attention_mask = batch.attention_mask.to(device, non_blocking=True)

                masked_input, mlm_labels = _make_mlm_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_id=vocab.mask_id,
                    pad_id=vocab.pad_id,
                    special_ids={vocab.usr_id, vocab.evt_id},
                    mask_prob=mask_prob,
                )

                logits = model(masked_input, attention_mask=attention_mask, return_mlm_logits=True)
                loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if is_main:
                    pbar.update(1)
                    pbar.set_postfix(train_loss=f"{float(loss.item()):.4f}")

                if (step + 1) % log_every == 0:
                    if _is_distributed():
                        dist.barrier()
                    model.eval()
                    if is_main and valid_loader is not None:
                        valid_loss = _evaluate(
                            model=model,
                            valid_loader=valid_loader,
                            loss_fn=loss_fn,
                            vocab=vocab,
                            mask_prob=mask_prob,
                            device=device,
                        )
                        if valid_loss < best_valid:
                            best_valid = valid_loss
                            save_checkpoint(
                                out_dir / "best.ckpt",
                                {
                                    "task": "mlm",
                                    "model_state": _unwrap_model(model).state_dict(),
                                    "model_cfg": cfg.__dict__,
                                    "tokenizer_dir": str(tokenizer_dir),
                                    "best_valid_loss": best_valid,
                                    "step": step + 1,
                                },
                            )
                        pbar.set_postfix(train_loss=f"{float(loss.item()):.4f}", valid_loss=f"{valid_loss:.4f}")
                    if _is_distributed():
                        dist.barrier()
                    model.train()

                step += 1
            epoch += 1
    finally:
        pbar.close()
        _cleanup_distributed()


if __name__ == "__main__":
    main()
