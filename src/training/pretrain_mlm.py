from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import time
from pathlib import Path
import urllib.request

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import src.model.pragma_lite.model as pragma_model_module
from src.common.yaml_utils import load_yaml
from src.model.pragma_lite.model import PragmaLiteConfig, PragmaLiteModel
from src.tokenizer.masking import MaskedEventCollator
from src.tokenizer.vocab import TokenizerVocab
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import load_tokenized_manifest_split, load_tokenized_split, set_seed


def _guess_tokenizer_dir(data_dir: Path) -> Path:
    candidate = data_dir.parent / "tokenizer"
    return candidate


def _guess_tokenizer_dir_from_manifest(manifest_path: Path) -> Path:
    manifest = load_yaml(manifest_path) if manifest_path.suffix in {".yaml", ".yml"} else json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_dir = manifest.get("tokenizer_dir")
    if not tokenizer_dir:
        raise ValueError(f"Manifest missing tokenizer_dir: {manifest_path}")
    return Path(str(tokenizer_dir))


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
    env_path = Path(os.environ.get("PRAGMA_DEBUG_ENV_FILE", ".dbg/pretrain-slow.env"))
    url = "http://127.0.0.1:7777/event"
    session_id = "pretrain-slow"
    run_id = os.environ.get("PRAGMA_DEBUG_RUN_ID", "pre-fix")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                url = line.split("=", 1)[1].strip() or url
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip() or session_id
    if not url:
        return
    hypothesis_id = str(payload.pop("hypothesis_id", "A"))
    location = str(payload.pop("location", "src/training/pretrain_mlm.py"))
    msg = str(payload.pop("msg", f"[DEBUG] {event}"))
    body = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": {
            "event": event,
            "rank": _rank(),
            **payload,
        },
        "ts": int(time.time() * 1000),
    }
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.25,
        ).read()
    except Exception:
        return


def _resolve_precision(train_cfg: dict[str, object], device: torch.device) -> str:
    training_cfg = train_cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}
    requested = str(training_cfg.get("precision", os.environ.get("PRECISION", "bf16"))).lower()
    if requested not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {requested}. Expected one of: fp32, bf16")
    if requested == "bf16" and device.type != "cuda":
        return "fp32"
    return requested


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if _env_bool("ENABLE_TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _resolve_dataloader_cfg(train_cfg: dict[str, object]) -> dict[str, object]:
    dataloader_cfg = train_cfg.get("dataloader", {})
    if not isinstance(dataloader_cfg, dict):
        dataloader_cfg = {}
    training_cfg = train_cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}
    num_workers = _env_int("DATALOADER_NUM_WORKERS", int(dataloader_cfg.get("num_workers", training_cfg.get("num_workers", 0))))
    pin_memory = _env_bool("DATALOADER_PIN_MEMORY", bool(dataloader_cfg.get("pin_memory", training_cfg.get("pin_memory", torch.cuda.is_available()))))
    persistent_workers = _env_bool("DATALOADER_PERSISTENT_WORKERS", bool(dataloader_cfg.get("persistent_workers", num_workers > 0)))
    prefetch_factor_raw = os.environ.get("DATALOADER_PREFETCH_FACTOR", dataloader_cfg.get("prefetch_factor"))
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }


def _resolve_scheduler_cfg(train_cfg: dict[str, object], max_steps: int) -> dict[str, object]:
    training_cfg = train_cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}
    scheduler_name = str(os.environ.get("SCHEDULER", training_cfg.get("scheduler", "constant"))).strip().lower()
    if scheduler_name not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Expected one of: constant, cosine")
    warmup_steps = _env_int("WARMUP_STEPS", int(training_cfg.get("warmup_steps", 0)))
    warmup_steps = max(0, min(warmup_steps, max(max_steps - 1, 0)))
    return {
        "name": scheduler_name,
        "warmup_steps": warmup_steps,
    }


def _lr_scale_for_update(
    scheduler_name: str,
    update_step: int,
    *,
    max_steps: int,
    warmup_steps: int,
) -> float:
    if update_step <= 0:
        return 1.0
    if warmup_steps > 0 and update_step <= warmup_steps:
        return float(update_step) / float(warmup_steps)
    if scheduler_name == "constant":
        return 1.0
    decay_steps = max(max_steps - warmup_steps, 1)
    clamped_step = min(max(update_step - warmup_steps, 0), decay_steps)
    progress = float(clamped_step) / float(decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _require_lmdb_for_full_scale(data_dir: Path) -> None:
    data_dir_str = str(data_dir).lower()
    if "full" not in data_dir_str:
        return
    lmdb_candidates = (
        data_dir / "dataset.lmdb",
        data_dir / "train.lmdb",
    )
    if any(path.exists() for path in lmdb_candidates):
        return
    raise RuntimeError(
        f"Full-scale training requires LMDB backend. Expected LMDB artifacts under {data_dir}."
    )


def _build_data_loaders(
    *,
    data_dir: Path,
    split_dir: Path,
    manifest_path: Path | None,
    train_split_name: str,
    valid_split_name: str,
    batch_size: int,
    seed: int,
    dataloader_cfg: dict[str, object],
    train_collator: MaskedEventCollator,
    valid_collator: MaskedEventCollator,
) -> tuple[object, object, DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    if manifest_path is not None:
        train_ds = load_tokenized_manifest_split(manifest_path, train_split_name)
        valid_ds = load_tokenized_manifest_split(manifest_path, valid_split_name)
    else:
        train_ds = load_tokenized_split(data_dir, train_split_name, split_dir=split_dir)
        valid_ds = load_tokenized_split(data_dir, valid_split_name, split_dir=split_dir)

    num_workers = int(dataloader_cfg["num_workers"])
    pin_memory = bool(dataloader_cfg["pin_memory"])
    persistent_workers = bool(dataloader_cfg["persistent_workers"])
    prefetch_factor = dataloader_cfg["prefetch_factor"]
    train_sampler = None
    valid_sampler = None
    if _is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=valid_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_ds, valid_ds, train_loader, valid_loader, train_sampler, valid_sampler


def _data_source_signature(data_dir: Path, manifest_path: Path | None) -> tuple[float, int]:
    if manifest_path is None:
        dataset_lmdb = data_dir / "train.lmdb"
        dataset_parquet = data_dir / "train.parquet"
        candidate = dataset_lmdb if dataset_lmdb.exists() else dataset_parquet
        stat = candidate.stat() if candidate.exists() else data_dir.stat()
        return (stat.st_mtime, int(stat.st_size))
    stat = manifest_path.stat()
    return (stat.st_mtime, int(stat.st_size))


def _count_ready_shards(manifest_path: Path | None) -> int | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    shards = manifest.get("shards", [])
    return sum(1 for entry in shards if isinstance(entry, dict) and str(entry.get("status", "ready")) == "ready")


def _append_metrics(metrics_path: Path, payload: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _maybe_generate_plots(metrics_path: Path, output_dir: Path, title_prefix: str) -> None:
    try:
        from tools.plot_pretrain_metrics import generate_plots, load_metrics
    except Exception:
        return
    if not metrics_path.exists():
        return
    try:
        entries = load_metrics(metrics_path)
        if entries:
            generate_plots(entries, output_dir, title_prefix)
    except Exception:
        return


def _masked_accuracy_stats(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    with torch.no_grad():
        valid = labels.ne(-100)
        count = int(valid.sum().item())
        if count == 0:
            return 0, 0
        preds = logits.argmax(dim=-1)
        correct = int(preds.eq(labels).logical_and(valid).sum().item())
    return correct, count


def _has_supervised_mlm_targets(labels: torch.Tensor) -> bool:
    return bool(labels.ne(-100).any().item())


def _supervised_target_rank_count(has_targets_local: bool, device: torch.device) -> int:
    if not _is_distributed():
        return int(has_targets_local)
    stats = torch.tensor([int(has_targets_local)], dtype=torch.int64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return int(stats.item())


def _iter_ddp_stride_probe_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    tracked: list[tuple[str, nn.Parameter]] = []
    for name, param in _unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(("profile_cls", "event_cls")):
            tracked.append((name, param))
            continue
        if param.ndim == 3 and tuple(param.shape[:2]) == (1, 1):
            tracked.append((name, param))
    return tracked


def _debug_tensor_layout(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _emit_ddp_stride_layout_snapshot(model: nn.Module, *, tag: str) -> None:
    source_file = Path(str(pragma_model_module.__file__)).resolve()
    try:
        source_text = source_file.read_text(encoding="utf-8")
    except Exception:
        source_text = ""
    for name, param in _iter_ddp_stride_probe_parameters(model):
        _debug_event(
            "ddp_param_layout_snapshot",
            hypothesis_id="D",
            location="src/training/pretrain_mlm.py:ddp_param_layout_snapshot",
            msg="[DEBUG] DDP stride probe parameter snapshot",
            tag=tag,
            parameter=name,
            param_layout=_debug_tensor_layout(param),
            model_source_file=str(source_file),
            model_source_mtime_ns=source_file.stat().st_mtime_ns if source_file.exists() else None,
            profile_repeat_present="self.profile_cls.repeat(" in source_text,
            event_repeat_present="self.event_cls.repeat(" in source_text,
            local_rank=os.environ.get("LOCAL_RANK"),
            world_size=os.environ.get("WORLD_SIZE"),
        )


def _register_ddp_stride_debug_hooks(model: nn.Module) -> None:
    max_events = _env_int("DDP_STRIDE_DEBUG_MAX_EVENTS", 6)
    hook_counts: dict[str, int] = {}
    for name, param in _iter_ddp_stride_probe_parameters(model):
        hook_counts[name] = 0

        def _hook(grad: torch.Tensor, *, parameter_name: str = name, parameter: nn.Parameter = param) -> torch.Tensor:
            mismatch = tuple(grad.stride()) != tuple(parameter.stride())
            seen = hook_counts[parameter_name]
            if not mismatch and seen >= max_events:
                return grad
            hook_counts[parameter_name] = seen + 1
            _debug_event(
                "ddp_grad_layout",
                hypothesis_id="C" if mismatch else "E",
                location="src/training/pretrain_mlm.py:ddp_grad_hook",
                msg="[DEBUG] DDP grad layout observed",
                parameter=parameter_name,
                mismatch=mismatch,
                grad_layout=_debug_tensor_layout(grad),
                param_layout=_debug_tensor_layout(parameter),
                observation_index=hook_counts[parameter_name],
                local_rank=os.environ.get("LOCAL_RANK"),
                world_size=os.environ.get("WORLD_SIZE"),
            )
            return grad

        param.register_hook(_hook)


def _emit_ddp_stride_grad_snapshot(model: nn.Module, *, step: int, epoch: int) -> None:
    for name, param in _iter_ddp_stride_probe_parameters(model):
        grad = param.grad
        mismatch = grad is not None and tuple(grad.stride()) != tuple(param.stride())
        _debug_event(
            "ddp_grad_snapshot",
            hypothesis_id="C" if mismatch else "E",
            location="src/training/pretrain_mlm.py:ddp_grad_snapshot",
            msg="[DEBUG] DDP grad snapshot after backward",
            step=step,
            epoch=epoch,
            parameter=name,
            mismatch=bool(mismatch),
            grad_layout=_debug_tensor_layout(grad),
            param_layout=_debug_tensor_layout(param),
            local_rank=os.environ.get("LOCAL_RANK"),
            world_size=os.environ.get("WORLD_SIZE"),
        )


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in _unwrap_model(model).parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().data.norm(2).item())
        total += grad_norm * grad_norm
    return total ** 0.5


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: PragmaLiteConfig,
    tokenizer_dir: Path,
    precision: str,
    best_valid: float,
    step: int,
    epoch: int,
) -> dict[str, object]:
    return {
        "task": "mlm",
        "model_state": _unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_cfg": cfg.__dict__,
        "tokenizer_dir": str(tokenizer_dir),
        "precision": precision,
        "best_valid_loss": best_valid,
        "step": step,
        "epoch": epoch,
    }


def _evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    precision: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    total_loss = 0.0
    total_batches = 0
    total_masked_correct = 0.0
    total_masked_count = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            if max_batches is not None and total_batches >= max_batches:
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
            if not _has_supervised_mlm_targets(mlm_labels):
                continue
            with _autocast_context(device, precision):
                logits = model(**model_inputs, return_mlm_logits=True)
                loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
            total_loss += float(loss.item())
            total_batches += 1
            masked_correct, masked_count = _masked_accuracy_stats(logits, mlm_labels)
            total_masked_correct += float(masked_correct)
            total_masked_count += float(masked_count)
    if _is_distributed():
        stats = torch.tensor(
            [total_loss, float(total_batches), total_masked_correct, total_masked_count],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_batches = int(stats[1].item())
        total_masked_correct = float(stats[2].item())
        total_masked_count = float(stats[3].item())
    valid_loss = total_loss / max(total_batches, 1) if total_batches else float("inf")
    valid_masked_accuracy = total_masked_correct / max(total_masked_count, 1.0)
    valid_perplexity = float(math.exp(min(valid_loss, 20.0))) if math.isfinite(valid_loss) else float("inf")
    return {
        "valid_loss": valid_loss,
        "valid_masked_accuracy": valid_masked_accuracy,
        "valid_perplexity": valid_perplexity,
        "valid_batches": float(total_batches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_dir")
    parser.add_argument("--manifest_path")
    parser.add_argument("--resume_from")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)["model"]
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    seed = int(train_cfg["training"].get("seed", 42))
    set_seed(seed)
    device, local_rank = _setup_device(args.device, args.backend)
    is_main = _is_main_process()
    precision = _resolve_precision(train_cfg, device)
    _configure_cuda_runtime(device)

    if args.tokenizer_dir:
        tokenizer_dir = Path(args.tokenizer_dir)
    elif manifest_path is not None:
        tokenizer_dir = _guess_tokenizer_dir_from_manifest(manifest_path)
    else:
        tokenizer_dir = _guess_tokenizer_dir(data_dir)
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
        calendar_mlp=bool(model_cfg.get("calendar_mlp", True)),
        calendar_hidden_dim=(
            int(model_cfg["calendar_hidden_dim"])
            if model_cfg.get("calendar_hidden_dim") is not None
            else None
        ),
        tie_mlm_to_embedding=bool(model_cfg.get("tie_mlm_to_embedding", True)),
    )
    if manifest_path is None:
        _require_lmdb_for_full_scale(data_dir)
    # #region debug-point A:dataset-init
    dataset_load_started = time.perf_counter()
    # #endregion
    model: nn.Module = PragmaLiteModel(cfg).to(device)
    # #region debug-point C:ddp-stride-probe
    _emit_ddp_stride_layout_snapshot(model, tag="before_ddp_wrap")
    _register_ddp_stride_debug_hooks(model)
    # #endregion
    if _is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    # #region debug-point C:ddp-stride-probe
    _emit_ddp_stride_layout_snapshot(model, tag="after_ddp_wrap")
    # #endregion

    batch_size = _env_int("TRAIN_BATCH_SIZE", int(train_cfg["training"].get("batch_size", 32)))
    data_cfg = train_cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    split_mode = str(os.environ.get("DATA_SPLIT_MODE", data_cfg.get("split_mode", "random"))).strip().lower()
    train_split_name = str(os.environ.get("TRAIN_SPLIT_NAME", data_cfg.get("train_split", "train"))).strip()
    valid_split_name = str(os.environ.get("VALID_SPLIT_NAME", data_cfg.get("valid_split", "valid"))).strip()
    dataloader_cfg = _resolve_dataloader_cfg(train_cfg)
    lr = float(train_cfg["training"].get("learning_rate", 3e-4))
    wd = float(train_cfg["training"].get("weight_decay", 0.01))
    max_steps = _env_int("MAX_STEPS", int(train_cfg["training"].get("max_steps", 1000)))
    scheduler_cfg = _resolve_scheduler_cfg(train_cfg, max_steps=max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    log_every = _env_int("LOG_EVERY", int(train_cfg["training"].get("log_every", 50)))
    valid_every = _env_int("VALID_EVERY", int(train_cfg["training"].get("valid_every", log_every)))
    full_valid_every = _env_int("FULL_VALID_EVERY", int(train_cfg["training"].get("full_valid_every", 0)))
    max_valid_batches_raw = _env_int("MAX_VALID_BATCHES", int(train_cfg["training"].get("max_valid_batches", 0)))
    max_valid_batches = max_valid_batches_raw if max_valid_batches_raw > 0 else None
    checkpoint_every = _env_int("CHECKPOINT_EVERY", log_every)
    plot_every = _env_int("PLOT_EVERY", int(train_cfg["training"].get("plot_every", 5000)))
    plots_dir = Path(os.environ.get("PLOTS_DIR", str(out_dir / "plots")))
    plot_title_prefix = str(os.environ.get("PLOT_TITLE_PREFIX", "Pretrain"))
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
    (
        train_ds,
        valid_ds,
        train_loader,
        valid_loader,
        train_sampler,
        valid_sampler,
    ) = _build_data_loaders(
        data_dir=data_dir,
        split_dir=split_dir,
        manifest_path=manifest_path,
        train_split_name=train_split_name,
        valid_split_name=valid_split_name,
        batch_size=batch_size,
        seed=seed,
        dataloader_cfg=dataloader_cfg,
        train_collator=train_collator,
        valid_collator=valid_collator,
    )
    num_workers = int(dataloader_cfg["num_workers"])
    pin_memory = bool(dataloader_cfg["pin_memory"])
    persistent_workers = bool(dataloader_cfg["persistent_workers"])
    prefetch_factor = dataloader_cfg["prefetch_factor"]
    current_data_signature = _data_source_signature(data_dir, manifest_path)
    # #region debug-point A:dataset-init
    _debug_event(
        "dataset_ready",
        hypothesis_id="A",
        location="src/training/pretrain_mlm.py:184",
        msg="[DEBUG] datasets loaded",
        data_dir=str(data_dir),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        split_mode=split_mode,
        train_split_name=train_split_name,
        valid_split_name=valid_split_name,
        train_len=len(train_ds),
        valid_len=len(valid_ds),
        train_backend=type(train_ds).__name__,
        valid_backend=type(valid_ds).__name__,
        elapsed_s=round(time.perf_counter() - dataset_load_started, 4),
    )
    # #endregion
    # #region debug-point B:loader-config
    _debug_event(
        "loader_config",
        hypothesis_id="B",
        location="src/training/pretrain_mlm.py:232",
        msg="[DEBUG] dataloader configured",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        enable_tf32=_env_bool("ENABLE_TF32", True),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        precision=precision,
    )
    # #endregion

    best_valid = float("inf")
    step = 0
    model.train()
    epoch = 0
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        _unwrap_model(model).load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        best_valid = float(checkpoint.get("best_valid_loss", best_valid))
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        _debug_event(
            "resume_loaded",
            hypothesis_id="A",
            location="src/training/pretrain_mlm.py:resume",
            msg="[DEBUG] resume checkpoint loaded",
            resume_from=str(resume_path),
            step=step,
            epoch=epoch,
            best_valid=best_valid,
        )
    next_update_step = min(step + 1, max_steps) if max_steps > 0 else 1
    scheduled_lr = lr * _lr_scale_for_update(
        str(scheduler_cfg["name"]),
        next_update_step,
        max_steps=max_steps,
        warmup_steps=int(scheduler_cfg["warmup_steps"]),
    )
    _set_optimizer_lr(optimizer, scheduled_lr)
    pbar = tqdm(total=max_steps, desc="pretrain", disable=not is_main, initial=step)
    last_step_finished_at = time.perf_counter()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            refreshed_signature = _data_source_signature(data_dir, manifest_path)
            if refreshed_signature != current_data_signature:
                (
                    train_ds,
                    valid_ds,
                    train_loader,
                    valid_loader,
                    train_sampler,
                    valid_sampler,
                ) = _build_data_loaders(
                    data_dir=data_dir,
                    split_dir=split_dir,
                    manifest_path=manifest_path,
                    train_split_name=train_split_name,
                    valid_split_name=valid_split_name,
                    batch_size=batch_size,
                    seed=seed,
                    dataloader_cfg=dataloader_cfg,
                    train_collator=train_collator,
                    valid_collator=valid_collator,
                )
                current_data_signature = refreshed_signature
                _debug_event(
                    "dataset_reloaded",
                    hypothesis_id="A",
                    location="src/training/pretrain_mlm.py:reload",
                    msg="[DEBUG] datasets reloaded from source",
                    train_len=len(train_ds),
                    valid_len=len(valid_ds),
                    manifest_path=str(manifest_path) if manifest_path is not None else None,
                    epoch=epoch,
                )
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= max_steps:
                    break
                # #region debug-point B:step-timing
                step_started_at = time.perf_counter()
                data_wait_s = step_started_at - last_step_finished_at
                if step < 8:
                    entity_preview = batch["entity_id"][: min(4, batch["entity_id"].shape[0])].detach().cpu().tolist()
                    active_token_count = int(batch["event_token_mask"].sum().item())
                    _debug_event(
                        "batch_loaded",
                        hypothesis_id="A",
                        location="src/training/pretrain_mlm.py:batch_loaded",
                        msg="[DEBUG] batch fetched from dataloader",
                        step=step + 1,
                        epoch=epoch,
                        rank=_rank(),
                        batch_size=int(batch["entity_id"].shape[0]),
                        entity_preview=entity_preview,
                        active_token_count=active_token_count,
                        data_wait_s=round(data_wait_s, 4),
                    )
                # #endregion
                # #region debug-point B:h2d
                h2d_started_at = time.perf_counter()
                # #endregion
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
                has_targets_local = _has_supervised_mlm_targets(mlm_labels)
                supervised_rank_count = _supervised_target_rank_count(has_targets_local, device)
                if supervised_rank_count == 0:
                    if _rank() == 0:
                        print(
                            f"[metrics][train] step={step + 1} skipped=no_masked_targets "
                            f"ready_shards={len(ready_train_shards)}"
                        )
                    continue
                # #region debug-point B:h2d
                h2d_s = time.perf_counter() - h2d_started_at
                if step < 8:
                    _debug_event(
                        "batch_h2d_done",
                        hypothesis_id="A",
                        location="src/training/pretrain_mlm.py:h2d_done",
                        msg="[DEBUG] batch copied to device",
                        step=step + 1,
                        epoch=epoch,
                        rank=_rank(),
                        h2d_s=round(h2d_s, 4),
                        profile_shape=list(model_inputs["profile_key_ids"].shape),
                        event_shape=list(model_inputs["event_key_ids"].shape),
                        label_shape=list(mlm_labels.shape),
                    )
                # #endregion
                # #region debug-point B:forward
                forward_started_at = time.perf_counter()
                # #endregion
                with _autocast_context(device, precision):
                    logits = model(**model_inputs, return_mlm_logits=True)
                    if has_targets_local:
                        loss = loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
                    else:
                        # Keep all ranks on the same DDP execution path when only a subset
                        # of local batches contains masked MLM targets.
                        loss = logits.sum() * 0.0
                masked_correct, masked_count = _masked_accuracy_stats(logits, mlm_labels)
                # #region debug-point B:forward
                forward_s = time.perf_counter() - forward_started_at
                if step < 8:
                    _debug_event(
                        "forward_done",
                        hypothesis_id="B",
                        location="src/training/pretrain_mlm.py:forward_done",
                        msg="[DEBUG] forward completed",
                        step=step + 1,
                        epoch=epoch,
                        rank=_rank(),
                        forward_s=round(forward_s, 4),
                        loss=float(loss.item()),
                        masked_count=int(masked_count),
                        has_targets_local=has_targets_local,
                        supervised_rank_count=supervised_rank_count,
                    )
                # #endregion

                # #region debug-point B:backward
                backward_started_at = time.perf_counter()
                # #endregion
                current_lr = float(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if step < _env_int("DDP_STRIDE_DEBUG_STEPS", 4):
                    # #region debug-point C:ddp-stride-probe
                    _emit_ddp_stride_grad_snapshot(model, step=step + 1, epoch=epoch)
                    # #endregion
                grad_norm = _grad_norm(model)
                # #region debug-point B:backward
                backward_s = time.perf_counter() - backward_started_at
                if step < 8:
                    _debug_event(
                        "backward_done",
                        hypothesis_id="B",
                        location="src/training/pretrain_mlm.py:backward_done",
                        msg="[DEBUG] backward completed",
                        step=step + 1,
                        epoch=epoch,
                        rank=_rank(),
                        backward_s=round(backward_s, 4),
                        grad_norm=round(float(grad_norm), 4),
                    )
                step_optimizer_started_at = time.perf_counter()
                # #endregion
                optimizer.step()
                next_lr = lr * _lr_scale_for_update(
                    str(scheduler_cfg["name"]),
                    min(step + 2, max_steps) if max_steps > 0 else 1,
                    max_steps=max_steps,
                    warmup_steps=int(scheduler_cfg["warmup_steps"]),
                )
                _set_optimizer_lr(optimizer, next_lr)
                # #region debug-point B:optimizer
                optimizer_s = time.perf_counter() - step_optimizer_started_at
                total_step_s = time.perf_counter() - step_started_at
                if step < 8:
                    _debug_event(
                        "optimizer_done",
                        hypothesis_id="B",
                        location="src/training/pretrain_mlm.py:optimizer_done",
                        msg="[DEBUG] optimizer step completed",
                        step=step + 1,
                        epoch=epoch,
                        rank=_rank(),
                        optimizer_s=round(optimizer_s, 4),
                        total_step_s=round(total_step_s, 4),
                        next_lr=float(next_lr),
                    )
                if step < 5 or (step + 1) % log_every == 0:
                    active_token_count = int(batch["event_token_mask"].sum().item())
                    gpu_allocated_gb = float(torch.cuda.memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
                    gpu_reserved_gb = float(torch.cuda.memory_reserved(device) / (1024**3)) if device.type == "cuda" else 0.0
                    summary_tensor = torch.tensor(
                        [
                            float(loss.item()),
                            float(masked_correct),
                            float(masked_count),
                            float(batch["entity_id"].shape[0]),
                            float(active_token_count),
                            float(data_wait_s),
                            float(h2d_s),
                            float(forward_s),
                            float(backward_s),
                            float(optimizer_s),
                            float(total_step_s),
                            float(grad_norm),
                            float(gpu_allocated_gb),
                            float(gpu_reserved_gb),
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    max_step_tensor = torch.tensor([float(total_step_s)], dtype=torch.float64, device=device)
                    if _is_distributed():
                        dist.all_reduce(summary_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(max_step_tensor, op=dist.ReduceOp.MAX)
                    train_loss_value = float(summary_tensor[0].item()) / float(world_size)
                    total_masked_correct = float(summary_tensor[1].item())
                    total_masked_count = float(summary_tensor[2].item())
                    global_batch_size = int(summary_tensor[3].item())
                    global_token_count = int(summary_tensor[4].item())
                    data_wait_avg = float(summary_tensor[5].item()) / float(world_size)
                    h2d_avg = float(summary_tensor[6].item()) / float(world_size)
                    forward_avg = float(summary_tensor[7].item()) / float(world_size)
                    backward_avg = float(summary_tensor[8].item()) / float(world_size)
                    optimizer_avg = float(summary_tensor[9].item()) / float(world_size)
                    total_step_max = float(max_step_tensor[0].item())
                    grad_norm_avg = float(summary_tensor[11].item()) / float(world_size)
                    gpu_allocated_avg = float(summary_tensor[12].item()) / float(world_size)
                    gpu_reserved_avg = float(summary_tensor[13].item()) / float(world_size)
                    masked_accuracy = total_masked_correct / max(total_masked_count, 1.0)
                    steps_per_sec = 1.0 / max(total_step_max, 1e-8)
                    samples_per_sec = float(global_batch_size) / max(total_step_max, 1e-8)
                    tokens_per_sec = float(global_token_count) / max(total_step_max, 1e-8)
                    ready_shards = _count_ready_shards(manifest_path)
                    _debug_event(
                        "train_step_timing",
                        hypothesis_id="B",
                        location="src/training/pretrain_mlm.py:265",
                        msg="[DEBUG] train step timing",
                        step=step + 1,
                        epoch=epoch,
                        batch_size=global_batch_size,
                        data_wait_s=round(data_wait_avg, 4),
                        h2d_s=round(h2d_avg, 4),
                        forward_s=round(forward_avg, 4),
                        backward_s=round(backward_avg, 4),
                        optimizer_s=round(optimizer_avg, 4),
                        total_step_s=round(total_step_max, 4),
                        masked_accuracy=round(masked_accuracy, 4),
                        steps_per_sec=round(steps_per_sec, 4),
                        ready_shards=ready_shards,
                    )
                    if is_main:
                        train_metrics = {
                            "kind": "train",
                            "step": step + 1,
                            "epoch": epoch,
                            "train_loss": train_loss_value,
                            "masked_accuracy": masked_accuracy,
                            "data_wait_s": data_wait_avg,
                            "h2d_s": h2d_avg,
                            "forward_s": forward_avg,
                            "backward_s": backward_avg,
                            "optimizer_s": optimizer_avg,
                            "total_step_s": total_step_max,
                            "steps_per_sec": steps_per_sec,
                            "samples_per_sec": samples_per_sec,
                            "tokens_per_sec": tokens_per_sec,
                            "grad_norm": grad_norm_avg,
                            "learning_rate": current_lr,
                            "gpu_mem_allocated_gb": gpu_allocated_avg,
                            "gpu_mem_reserved_gb": gpu_reserved_avg,
                            "num_ready_shards": ready_shards,
                            "global_batch_size": global_batch_size,
                            "global_token_count": global_token_count,
                        }
                        _append_metrics(metrics_path, train_metrics)
                        if plot_every > 0 and (step + 1) % plot_every == 0:
                            _maybe_generate_plots(metrics_path, plots_dir, plot_title_prefix)
                        print(
                            "[metrics][train] "
                            f"step={step + 1} loss={train_loss_value:.4f} "
                            f"masked_acc={masked_accuracy:.4f} "
                            f"steps_per_sec={steps_per_sec:.2f} "
                            f"gpu_mem_gb={gpu_allocated_avg:.2f} "
                            f"ready_shards={ready_shards}",
                            flush=True,
                        )
                last_step_finished_at = time.perf_counter()
                # #endregion

                if is_main:
                    pbar.update(1)
                    pbar.set_postfix(train_loss=f"{float(loss.item()):.4f}")
                    if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                        save_checkpoint(
                            out_dir / "last.ckpt",
                            _checkpoint_payload(
                                model=model,
                                optimizer=optimizer,
                                cfg=cfg,
                                tokenizer_dir=tokenizer_dir,
                                precision=precision,
                                best_valid=best_valid,
                                step=step + 1,
                                epoch=epoch,
                            ),
                        )

                if valid_every > 0 and (step + 1) % valid_every == 0:
                    run_full_validation = full_valid_every > 0 and (step + 1) % full_valid_every == 0
                    eval_mode = "full" if run_full_validation or max_valid_batches is None else "quick"
                    eval_max_batches = None if eval_mode == "full" else max_valid_batches
                    _debug_event(
                        "pre_eval",
                        hypothesis_id="D",
                        location="src/training/pretrain_mlm.py:297",
                        msg="[DEBUG] entering evaluation window",
                        step=step + 1,
                        epoch=epoch,
                        loss=float(loss.item()),
                        eval_mode=eval_mode,
                        max_valid_batches=eval_max_batches,
                    )
                    model.eval()
                    _debug_event(
                        "eval_start",
                        hypothesis_id="D",
                        location="src/training/pretrain_mlm.py:301",
                        msg="[DEBUG] evaluation started",
                        step=step + 1,
                        epoch=epoch,
                        eval_mode=eval_mode,
                        max_valid_batches=eval_max_batches,
                    )
                    valid_metrics = _evaluate(
                        model=model,
                        valid_loader=valid_loader,
                        loss_fn=loss_fn,
                        device=device,
                        precision=precision,
                        max_batches=eval_max_batches,
                    )
                    valid_loss = float(valid_metrics["valid_loss"])
                    _debug_event(
                        "eval_end",
                        hypothesis_id="D",
                        location="src/training/pretrain_mlm.py:309",
                        msg="[DEBUG] evaluation completed",
                        step=step + 1,
                        epoch=epoch,
                        valid_loss=valid_loss,
                        eval_mode=eval_mode,
                        valid_batches=float(valid_metrics["valid_batches"]),
                    )
                    if is_main:
                        if eval_mode == "full" and valid_loss < best_valid:
                            best_valid = valid_loss
                            save_checkpoint(
                                out_dir / "best.ckpt",
                                _checkpoint_payload(
                                    model=model,
                                    optimizer=optimizer,
                                    cfg=cfg,
                                    tokenizer_dir=tokenizer_dir,
                                    precision=precision,
                                    best_valid=best_valid,
                                    step=step + 1,
                                    epoch=epoch,
                                ),
                            )
                        eval_record = {
                            "kind": "valid",
                            "step": step + 1,
                            "epoch": epoch,
                            "eval_mode": eval_mode,
                            "valid_loss": valid_loss,
                            "valid_masked_accuracy": float(valid_metrics["valid_masked_accuracy"]),
                            "valid_perplexity": float(valid_metrics["valid_perplexity"]),
                            "valid_batches": int(valid_metrics["valid_batches"]),
                            "best_valid_loss": best_valid,
                            "num_ready_shards": _count_ready_shards(manifest_path),
                        }
                        _append_metrics(metrics_path, eval_record)
                        if plot_every > 0 and (step + 1) % plot_every == 0:
                            _maybe_generate_plots(metrics_path, plots_dir, plot_title_prefix)
                        save_checkpoint(
                            out_dir / "last.ckpt",
                            _checkpoint_payload(
                                model=model,
                                optimizer=optimizer,
                                cfg=cfg,
                                tokenizer_dir=tokenizer_dir,
                                precision=precision,
                                best_valid=best_valid,
                                step=step + 1,
                                epoch=epoch,
                            ),
                        )
                        print(
                            "[metrics][valid] "
                            f"mode={eval_mode} "
                            f"step={step + 1} valid_loss={valid_loss:.4f} "
                            f"valid_masked_acc={float(valid_metrics['valid_masked_accuracy']):.4f} "
                            f"valid_ppl={float(valid_metrics['valid_perplexity']):.4f} "
                            f"batches={int(valid_metrics['valid_batches'])} "
                            f"best_valid={best_valid:.4f}",
                            flush=True,
                        )
                        pbar.set_postfix(
                            train_loss=f"{float(loss.item()):.4f}",
                            valid_loss=f"{valid_loss:.4f}",
                            valid_acc=f"{float(valid_metrics['valid_masked_accuracy']):.4f}",
                        )
                    model.train()

                step += 1
            epoch += 1
    finally:
        if is_main and step > 0:
            save_checkpoint(
                out_dir / "last.ckpt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    tokenizer_dir=tokenizer_dir,
                    precision=precision,
                    best_valid=best_valid,
                    step=step,
                    epoch=epoch,
                ),
            )
        pbar.close()
        _cleanup_distributed()


if __name__ == "__main__":
    main()
