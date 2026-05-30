from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"train_loss=([0-9]+(?:\.[0-9]+)?)")


def extract_losses(log_path: Path) -> list[float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return [float(match.group(1)) for match in LOSS_PATTERN.finditer(text)]


def plot_losses(losses: list[float], output_path: Path, title: str) -> None:
    if not losses:
        raise ValueError("No train_loss values found in log")
    steps = list(range(1, len(losses) + 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.8))
    plt.plot(steps, losses, linewidth=1.2, color="#2563eb")
    plt.xlabel("Logged Update")
    plt.ylabel("Train Loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--title", default="Training Loss")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_path = Path(args.output_file)
    losses = extract_losses(log_path)
    plot_losses(losses, output_path, args.title)
    print(f"saved plot: {output_path}")
    print(f"points: {len(losses)}")
    print(f"first_loss: {losses[0]:.4f}")
    print(f"last_loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
