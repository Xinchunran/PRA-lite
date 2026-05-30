from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.common.yaml_utils import load_yaml
from src.model.pragma_lite.model import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.masking import MaskedEventCollator
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import save_checkpoint
from src.training.data import load_tokenized_split, set_seed


def _guess_tokenizer_dir(data_dir: Path) -> Path:
    candidate = data_dir.parent / "tokenizer"
    return candidate


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _unwrap_model(model: nn.Module) -> nn.Module:
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


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _debug_event(event: str, **payload: object) -> None:
    debug_path = os.environ.get("PRAGMA_DDP_DEBUG_FILE")
    if not debug_path:
        return
    row = {
        "ts": time.time(),
        "event": event,
        "rank": _rank(),
        **payload,
    }
    Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(debug_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in valid_loader:
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key
                in {
                    "profile_key_ids",
                    "profile_value_ids",
                    "profile_value_pos",
                    "profile_time",
                    "profile_mask",
                    "event_key_ids",
                    "event_value_ids",
                    "event_value_pos",
                    "event_token_mask",
                    "event_time",
                    "calendar_features",
                    "event_mask",
                }
            }
            logits = model(**model_inputs, return_mlm_logits=True)
            mlm_labels = batch["mlm_labels"].to(device, non_blocking=True)
            loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
            total_loss += float(loss.item())
            total_batches += 1
    if _is_distributed():
        stats = torch.tensor([total_loss, float(total_batches)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_batches = int(stats[1].item())
    return total_loss / max(total_batches, 1) if total_batches else float("inf")


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
        d_ffn=int(model_cfg.get("d_ffn", int(model_cfg.get("d_model", 192)) * 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_profile_tokens=int(model_cfg.get("max_profile_tokens", 200)),
        max_event_tokens=int(model_cfg.get("max_event_tokens", 24)),
        max_events=int(model_cfg.get("max_events", 512)),
        profile_layers=int(model_cfg.get("profile_layers", 1)),
        event_layers=int(model_cfg.get("event_layers", int(model_cfg.get("n_layers", 4)))),
        history_layers=int(model_cfg.get("history_layers", int(model_cfg.get("n_layers", 4)))),
    )
    train_ds = load_tokenized_split(data_dir, "train", split_dir=split_dir)
    valid_ds = load_tokenized_split(data_dir, "valid", split_dir=split_dir)
    model: nn.Module = PragmaLiteModel(cfg).to(device)
    if _is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    batch_size = int(train_cfg["training"].get("batch_size", 32))
    num_workers = int(train_cfg["training"].get("num_workers", 0))
    pin_memory = bool(train_cfg["training"].get("pin_memory", torch.cuda.is_available()))
    train_sampler = None
    valid_sampler = None
    if _is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, seed=seed)
    lr = float(train_cfg["training"].get("learning_rate", 3e-4))
    wd = float(train_cfg["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    max_steps = int(train_cfg["training"].get("max_steps", 1000))
    log_every = int(train_cfg["training"].get("log_every", 50))
    masking_cfg = train_cfg.get("masking", {})
    train_collator = MaskedEventCollator(
        mask_token_id=vocab.mask_id,
        unk_token_id=vocab.unk_id,
        pad_token_id=vocab.pad_id,
        token_mask_probability=float(masking_cfg.get("token_mask_prob", 0.15)),
        event_mask_probability=float(masking_cfg.get("event_mask_prob", 0.10)),
        key_mask_probability=float(masking_cfg.get("key_mask_prob", 0.10)),
        unk_probability=float(masking_cfg.get("unk_prob", 0.10)),
        seed=seed,
    )
    valid_collator = MaskedEventCollator(
        mask_token_id=vocab.mask_id,
        unk_token_id=vocab.unk_id,
        pad_token_id=vocab.pad_id,
        token_mask_probability=float(masking_cfg.get("token_mask_prob", 0.15)),
        event_mask_probability=float(masking_cfg.get("event_mask_prob", 0.10)),
        key_mask_probability=float(masking_cfg.get("key_mask_prob", 0.10)),
        unk_probability=float(masking_cfg.get("unk_prob", 0.10)),
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=valid_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

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
                model_inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                    if key
                    in {
                        "profile_key_ids",
                        "profile_value_ids",
                        "profile_value_pos",
                        "profile_time",
                        "profile_mask",
                        "event_key_ids",
                        "event_value_ids",
                        "event_value_pos",
                        "event_token_mask",
                        "event_time",
                        "calendar_features",
                        "event_mask",
                    }
                }
                mlm_labels = batch["mlm_labels"].to(device, non_blocking=True)
                logits = model(**model_inputs, return_mlm_logits=True)
                loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if is_main:
                    pbar.update(1)
                    pbar.set_postfix(train_loss=f"{float(loss.item()):.4f}")

                if (step + 1) % log_every == 0:
                    _debug_event("pre_barrier_eval", step=step + 1, epoch=epoch, loss=float(loss.item()))
                    _debug_event("post_barrier_eval", step=step + 1, epoch=epoch)
                    model.eval()
                    _debug_event("eval_start", step=step + 1, epoch=epoch)
                    valid_loss = _evaluate(
                        model=model,
                        valid_loader=valid_loader,
                        loss_fn=loss_fn,
                        device=device,
                    )
                    _debug_event("eval_end", step=step + 1, epoch=epoch, valid_loss=valid_loss)
                    if is_main:
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
                    model.train()

                step += 1
            epoch += 1
    finally:
        pbar.close()
        _cleanup_distributed()


if __name__ == "__main__":
    main()
