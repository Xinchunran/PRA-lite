from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def _series(entries: list[dict[str, Any]], kind: str, metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for entry in entries:
        if str(entry.get("kind")) != kind:
            continue
        value = entry.get(metric)
        step = entry.get("step")
        if value is None or step is None:
            continue
        xs.append(int(step))
        ys.append(float(value))
    return xs, ys


def _single_plot(entries: list[dict[str, Any]], kind: str, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    xs, ys = _series(entries, kind, metric)
    if not xs:
        return
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(xs, ys, linewidth=1.3, color="#2563eb")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _multi_plot(
    entries: list[dict[str, Any]],
    kind: str,
    metrics: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plotted = False
    plt.figure(figsize=(10, 5))
    for metric, label, color in metrics:
        xs, ys = _series(entries, kind, metric)
        if not xs:
            continue
        plotted = True
        plt.plot(xs, ys, linewidth=1.2, label=label, color=color)
    if not plotted:
        plt.close()
        return
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def generate_plots(entries: list[dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _single_plot(entries, "train", "train_loss", output_dir / "train_loss.png", f"{title_prefix} Train Loss", "Loss")
    _single_plot(
        entries,
        "train",
        "masked_accuracy",
        output_dir / "train_masked_accuracy.png",
        f"{title_prefix} Train Masked Accuracy",
        "Accuracy",
    )
    _single_plot(entries, "valid", "valid_loss", output_dir / "valid_loss.png", f"{title_prefix} Valid Loss", "Loss")
    _single_plot(
        entries,
        "valid",
        "valid_masked_accuracy",
        output_dir / "valid_masked_accuracy.png",
        f"{title_prefix} Valid Masked Accuracy",
        "Accuracy",
    )
    _single_plot(
        entries,
        "valid",
        "valid_perplexity",
        output_dir / "valid_perplexity.png",
        f"{title_prefix} Valid Perplexity",
        "Perplexity",
    )
    _single_plot(entries, "train", "grad_norm", output_dir / "grad_norm.png", f"{title_prefix} Gradient Norm", "L2 Norm")
    _single_plot(
        entries,
        "train",
        "num_ready_shards",
        output_dir / "ready_shards.png",
        f"{title_prefix} Ready Shards",
        "Shard Count",
    )
    _single_plot(
        entries,
        "train",
        "learning_rate",
        output_dir / "learning_rate.png",
        f"{title_prefix} Learning Rate",
        "Learning Rate",
    )

    _multi_plot(
        entries,
        "train",
        [
            ("steps_per_sec", "steps/s", "#2563eb"),
            ("samples_per_sec", "samples/s", "#16a34a"),
            ("tokens_per_sec", "tokens/s", "#dc2626"),
        ],
        output_dir / "throughput.png",
        f"{title_prefix} Throughput",
        "Rate",
    )
    _multi_plot(
        entries,
        "train",
        [
            ("data_wait_s", "data_wait_s", "#7c3aed"),
            ("h2d_s", "h2d_s", "#2563eb"),
            ("forward_s", "forward_s", "#16a34a"),
            ("backward_s", "backward_s", "#dc2626"),
            ("optimizer_s", "optimizer_s", "#f59e0b"),
            ("total_step_s", "total_step_s", "#111827"),
        ],
        output_dir / "timings.png",
        f"{title_prefix} Step Timing",
        "Seconds",
    )
    _multi_plot(
        entries,
        "train",
        [
            ("gpu_mem_allocated_gb", "allocated_gb", "#2563eb"),
            ("gpu_mem_reserved_gb", "reserved_gb", "#dc2626"),
        ],
        output_dir / "gpu_memory.png",
        f"{title_prefix} GPU Memory",
        "GB",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title_prefix", default="Pretrain")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    entries = load_metrics(metrics_path)
    if not entries:
        raise ValueError("No metrics entries found")
    generate_plots(entries, output_dir, args.title_prefix)
    print(f"saved plots to: {output_dir}")
    print(f"entries: {len(entries)}")


if __name__ == "__main__":
    main()
