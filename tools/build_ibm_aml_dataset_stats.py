from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

mpl_cache_dir = Path(".matplotlib-cache")
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NATURE_TEAL = "#2a9d8f"
NATURE_BLUE = "#457b9d"
NATURE_CYAN = "#76c7c0"
NATURE_NAVY = "#1f5c7a"
NATURE_LIGHT = "#dff3f1"
PLOT_COLORS = [NATURE_BLUE, NATURE_TEAL, NATURE_CYAN, "#8ecae6", "#5fa8a0"]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _write_markdown_table(path: Path, title: str, frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    printable = frame.astype(object).where(frame.notna(), "")
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in printable.itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([f"# {title}", "", header, sep, *rows, ""]) + "\n", encoding="utf-8")


def _safe_rate(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def _mode_or_missing(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "missing"
    mode = non_null.mode(dropna=True)
    if mode.empty:
        return str(non_null.iloc[-1])
    return str(mode.iloc[0])


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    weights = weights.fillna(0.0)
    values = values.fillna(0.0)
    if float(weights.sum()) <= 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _classify_columns(frame: pd.DataFrame, skip: set[str]) -> dict[str, int]:
    numeric = 0
    categorical = 0
    textual = 0
    for column in frame.columns:
        if column in skip:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric += 1
            continue
        lower_name = column.lower()
        if "name" in lower_name or "text" in lower_name:
            textual += 1
        else:
            categorical += 1
    return {"numeric": numeric, "categorical": categorical, "textual": textual}


def _build_group_bias_table(
    labels: pd.DataFrame,
    group_values: pd.DataFrame,
    group_col: str,
    top_k: int = 8,
) -> pd.DataFrame:
    merged = labels.merge(group_values[["entity_id", group_col]], on="entity_id", how="left")
    merged[group_col] = merged[group_col].fillna("missing").astype("string")
    top_groups = merged[group_col].value_counts().head(top_k).index.tolist()
    filtered = merged[merged[group_col].isin(top_groups)].copy()
    summary = (
        filtered.groupby(group_col, dropna=False)["label"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"size": "entities", "sum": "positives", "mean": "positive_rate"})
        .sort_values(["positive_rate", "entities"], ascending=[False, False])
    )
    return summary


def _plot_bar(
    path: Path,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    color: str,
    rotate_xticks: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(frame[x_col].astype(str), frame[y_col].astype(float), color=color, width=0.72)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=rotate_xticks)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_field_types(path: Path, field_type_frame: pd.DataFrame) -> None:
    categories = ["numeric", "categorical", "textual"]
    labels = field_type_frame["source"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for idx, category in enumerate(categories):
        ax.bar(
            x + (idx - 1) * width,
            field_type_frame[category].astype(float),
            width=width,
            label=category,
            color=PLOT_COLORS[idx],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Count")
    ax.set_title("IBM AML Field Type Composition")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_encoder_inputs(path: Path, encoder_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    metrics = [
        ("history_length_mean", "Mean History Length", NATURE_BLUE),
        ("token_length_mean", "Mean Token Length", NATURE_TEAL),
        ("empty_history_rate", "Empty History Rate", NATURE_CYAN),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.hist(encoder_frame[col].astype(float), bins=24, color=color, edgecolor="white")
        if col == "history_length_mean":
            ax.axvline(256.0, color=NATURE_NAVY, linestyle="--", linewidth=1.2, label="max_events")
            ax.legend()
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_bias(path: Path, bank_bias: pd.DataFrame, payment_bias: pd.DataFrame, label_rate: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, frame, group_col, title, color in (
        (axes[0], bank_bias, "primary_bank", "Positive Rate By Primary Bank", NATURE_BLUE),
        (axes[1], payment_bias, "dominant_payment_format", "Positive Rate By Dominant Payment Format", NATURE_TEAL),
    ):
        ax.bar(frame[group_col].astype(str), frame["positive_rate"].astype(float), color=color, width=0.72)
        ax.axhline(label_rate, color=NATURE_NAVY, linestyle="--", linewidth=1.1, label="overall")
        ax.set_title(title)
        ax.set_ylabel("Positive Rate")
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_shard_eval_distribution(path: Path, manifest_frame: pd.DataFrame) -> None:
    plotted = manifest_frame.sort_values("num_eval_points", ascending=False).head(20).copy()
    colors = [NATURE_TEAL if name != "shard_00120" else "#e76f51" for name in plotted["shard_name"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.bar(plotted["shard_name"], plotted["num_eval_points"].astype(float), color=colors, width=0.72)
    ax.set_title("Largest Eval-Point Shards")
    ax.set_ylabel("Evaluation Points")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_encoder_frame(stream_root: Path, manifest: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shard in manifest.get("shards", []):
        if shard.get("status") != "ready":
            continue
        shard_name = str(shard["name"])
        summary_path = stream_root / "tokenized_shards" / shard_name / "shard_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        num_records = int(summary.get("num_records", 0))
        rows.append(
            {
                "shard_name": shard_name,
                "num_records": num_records,
                "history_length_mean": float(summary.get("history_length_mean", 0.0)),
                "history_length_max": float(summary.get("history_length_max", 0.0)),
                "token_length_mean": float(summary.get("token_length_mean", 0.0)),
                "batching_event_count_mean": float(summary.get("batching_event_count_mean", 0.0)),
                "batching_profile_token_count_mean": float(summary.get("batching_profile_token_count_mean", 0.0)),
                "empty_history_records": float(summary.get("empty_history_records", 0.0)),
                "max_events": float(summary.get("max_events", 0.0)),
                "max_event_tokens": float(summary.get("max_event_tokens", 0.0)),
                "max_profile_tokens": float(summary.get("max_profile_tokens", 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["empty_history_rate"] = frame["empty_history_records"] / frame["num_records"].clip(lower=1.0)
    return frame


def build_stats(processed_dir: Path, stream_root: Path, output_dir: Path) -> None:
    _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = _load_json(processed_dir / "schema.json")
    manifest = _load_json(stream_root / "manifest.json")
    split_summary = _load_json(stream_root / "split_summary.json")
    eval_point_summary = _load_json(stream_root / "eval_points" / "eval_point_sampling_summary.json")
    vocab_summary = _load_json(stream_root / "tokenizer" / "vocab_summary.json")

    labels = pd.read_parquet(processed_dir / "labels.parquet", columns=["entity_id", "label"])
    profiles = pd.read_parquet(processed_dir / "profiles.parquet")
    schema_event_columns = set(schema.get("event_columns", []))
    event_columns = ["entity_id"]
    for col in ("direction", "payment_format", "bank_id"):
        if col in schema_event_columns:
            event_columns.append(col)
    events = pd.read_parquet(processed_dir / "events.parquet", columns=event_columns)

    label_rate = float(labels["label"].mean())
    event_counts = events.groupby("entity_id").size().rename("event_count") if not events.empty else pd.Series(dtype="int64")
    events_per_entity_mean = float(event_counts.mean()) if not event_counts.empty else 0.0
    events_per_entity_p95 = float(event_counts.quantile(0.95)) if not event_counts.empty else 0.0
    events_per_entity_max = int(event_counts.max()) if not event_counts.empty else 0

    profile_type_counts = _classify_columns(profiles, {"entity_id"})
    event_type_counts = _classify_columns(events, {"entity_id"})
    tokenizer_type_counts = {
        "numeric": int(vocab_summary["field_value_type_counts"].get("numeric", 0)),
        "categorical": int(vocab_summary["field_value_type_counts"].get("categorical", 0)),
        "textual": int(vocab_summary["field_value_type_counts"].get("textual", 0)),
    }

    ready_shards = [shard for shard in manifest.get("shards", []) if shard.get("status") == "ready"]
    pending_shards = [shard for shard in manifest.get("shards", []) if shard.get("status") != "ready"]

    encoder_frame = _build_encoder_frame(stream_root, manifest)
    weighted_records = encoder_frame["num_records"] if not encoder_frame.empty else pd.Series(dtype="float64")
    encoder_summary = pd.DataFrame(
        [
            {
                "metric": "max_events_limit",
                "value": float(encoder_frame["max_events"].max()) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "max_event_tokens_limit",
                "value": float(encoder_frame["max_event_tokens"].max()) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "max_profile_tokens_limit",
                "value": float(encoder_frame["max_profile_tokens"].max()) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "weighted_history_length_mean",
                "value": _weighted_average(encoder_frame["history_length_mean"], weighted_records) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "history_length_max",
                "value": float(encoder_frame["history_length_max"].max()) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "weighted_token_length_mean",
                "value": _weighted_average(encoder_frame["token_length_mean"], weighted_records) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "weighted_batching_event_count_mean",
                "value": _weighted_average(encoder_frame["batching_event_count_mean"], weighted_records) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "weighted_profile_token_count_mean",
                "value": _weighted_average(encoder_frame["batching_profile_token_count_mean"], weighted_records) if not encoder_frame.empty else 0.0,
            },
            {
                "metric": "weighted_empty_history_rate",
                "value": _weighted_average(encoder_frame["empty_history_rate"], weighted_records) if not encoder_frame.empty else 0.0,
            },
        ]
    )

    manifest_frame = pd.DataFrame(
        [
            {
                "shard_name": str(shard["name"]),
                "status": str(shard.get("status", "unknown")),
                "num_records": int(shard.get("num_records", 0)),
                "num_eval_points": int(shard.get("num_eval_points", 0)),
                "num_unique_entities": int(shard.get("num_unique_entities", 0)),
            }
            for shard in manifest.get("shards", [])
        ]
    )

    dataset_overview = pd.DataFrame(
        [
            {"metric": "processed_entities", "value": int(labels["entity_id"].nunique())},
            {"metric": "processed_events", "value": int(len(events))},
            {"metric": "processed_profiles", "value": int(len(profiles))},
            {"metric": "positive_entities", "value": int(labels["label"].sum())},
            {"metric": "positive_rate", "value": label_rate},
            {"metric": "events_per_entity_mean", "value": events_per_entity_mean},
            {"metric": "events_per_entity_p95", "value": events_per_entity_p95},
            {"metric": "events_per_entity_max", "value": events_per_entity_max},
            {"metric": "total_eval_points", "value": int(eval_point_summary["num_eval_points"])},
            {"metric": "pretrain_eval_points", "value": int(eval_point_summary["num_pretrain_eval_points"])},
            {"metric": "downstream_eval_points", "value": int(eval_point_summary["num_downstream_eval_points"])},
            {"metric": "ready_shards", "value": int(len(ready_shards))},
            {"metric": "pending_shards", "value": int(len(pending_shards))},
            {"metric": "expected_shards", "value": int(manifest.get("expected_shards", 0))},
            {"metric": "vocab_size", "value": int(vocab_summary["vocab_size"])},
        ]
    )

    field_type_summary = pd.DataFrame(
        [
            {"source": "processed_profile", **profile_type_counts},
            {"source": "processed_event", **event_type_counts},
            {"source": "tokenizer_fields", **tokenizer_type_counts},
        ]
    )

    split_frame = pd.DataFrame(
        [
            {"split": split_name, "num_records": int(values["num_records"])}
            for split_name, values in split_summary["splits"].items()
        ]
    )
    split_frame["share"] = split_frame["num_records"] / split_frame["num_records"].sum()

    bank_bias = pd.DataFrame(columns=["primary_bank", "entities", "positives", "positive_rate"])
    if "primary_bank" in profiles.columns:
        bank_bias = _build_group_bias_table(labels, profiles[["entity_id", "primary_bank"]], "primary_bank")

    payment_bias = pd.DataFrame(columns=["dominant_payment_format", "entities", "positives", "positive_rate"])
    if "dominant_payment_format" in profiles.columns:
        payment_bias = _build_group_bias_table(
            labels,
            profiles[["entity_id", "dominant_payment_format"]],
            "dominant_payment_format",
        )

    bias_summary = pd.DataFrame(
        [
            {
                "metric": "overall_positive_rate",
                "value": label_rate,
            },
            {
                "metric": "class_imbalance_ratio_negative_to_positive",
                "value": _safe_rate(float((labels["label"] == 0).sum()), float((labels["label"] == 1).sum())),
            },
            {
                "metric": "primary_bank_positive_rate_gap_top_groups",
                "value": float(bank_bias["positive_rate"].max() - bank_bias["positive_rate"].min()) if not bank_bias.empty else 0.0,
            },
            {
                "metric": "dominant_payment_format_positive_rate_gap_top_groups",
                "value": float(payment_bias["positive_rate"].max() - payment_bias["positive_rate"].min()) if not payment_bias.empty else 0.0,
            },
        ]
    )

    dataset_overview.to_csv(output_dir / "dataset_overview.csv", index=False)
    field_type_summary.to_csv(output_dir / "field_type_summary.csv", index=False)
    encoder_summary.to_csv(output_dir / "encoder_input_summary.csv", index=False)
    split_frame.to_csv(output_dir / "split_record_summary.csv", index=False)
    bias_summary.to_csv(output_dir / "bias_summary.csv", index=False)
    manifest_frame.to_csv(output_dir / "manifest_shard_summary.csv", index=False)
    if not bank_bias.empty:
        bank_bias.to_csv(output_dir / "primary_bank_bias.csv", index=False)
    if not payment_bias.empty:
        payment_bias.to_csv(output_dir / "dominant_payment_format_bias.csv", index=False)

    _write_markdown_table(output_dir / "dataset_overview.md", "IBM AML Dataset Overview", dataset_overview)
    _write_markdown_table(output_dir / "field_type_summary.md", "IBM AML Field Type Summary", field_type_summary)
    _write_markdown_table(output_dir / "encoder_input_summary.md", "IBM AML Encoder Input Summary", encoder_summary)
    _write_markdown_table(output_dir / "split_record_summary.md", "IBM AML Split Record Summary", split_frame)
    _write_markdown_table(output_dir / "bias_summary.md", "IBM AML Bias Summary", bias_summary)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_grouped_field_types(plots_dir / "field_type_counts.png", field_type_summary)
    _plot_bar(
        plots_dir / "split_record_counts.png",
        split_frame,
        x_col="split",
        y_col="num_records",
        title="IBM AML Split Record Counts",
        ylabel="Records",
        color=NATURE_BLUE,
    )
    _plot_encoder_inputs(plots_dir / "encoder_input_distributions.png", encoder_frame)
    _plot_shard_eval_distribution(plots_dir / "largest_shard_eval_points.png", manifest_frame)
    if not bank_bias.empty and not payment_bias.empty:
        _plot_bias(plots_dir / "label_bias_groups.png", bank_bias, payment_bias, label_rate)

    report = {
        "processed_dir": str(processed_dir),
        "stream_root": str(stream_root),
        "output_dir": str(output_dir),
        "dataset_overview": dataset_overview.to_dict(orient="records"),
        "field_type_summary": field_type_summary.to_dict(orient="records"),
        "encoder_input_summary": encoder_summary.to_dict(orient="records"),
        "bias_summary": bias_summary.to_dict(orient="records"),
        "notes": [
            "Charts use a blue-teal palette suitable for publication-style figures.",
            "shard_00120 is highlighted in the largest-shard plot because it is currently the only pending shard.",
            f"Overall positive rate is {label_rate:.4f}; inspect the group disparity tables before claiming the dataset is unbiased.",
        ],
    }
    (output_dir / "dataset_stats_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed/ibm_aml_li_medium"))
    parser.add_argument("--stream_root", type=Path, default=Path("data/streaming/ibm_aml_li_medium_pragma_lite_full"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/ibm_aml_li_medium/stat_plot"))
    args = parser.parse_args()
    build_stats(args.processed_dir, args.stream_root, args.output_dir)


if __name__ == "__main__":
    main()
