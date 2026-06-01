from __future__ import annotations

import json
from pathlib import Path

from src.benchmarks.transxion_binary_benchmark import (
    build_raw_feature_table,
    run_transxion_binary_benchmark,
)
from tests.benchmark_fixtures import (
    write_checkpoint,
    write_processed_transxion_like_dir,
    write_splits,
    write_tokenized_dataset,
    write_tokenizer,
)


def test_build_raw_feature_table_aggregates_profiles_and_events(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    write_processed_transxion_like_dir(processed_dir)

    features = build_raw_feature_table(processed_dir, entity_ids={1, 2, 3})

    assert set(features["entity_id"]) == {1, 2, 3}
    assert "event_count" in features.columns
    assert "evt_amount_sum" in features.columns
    assert "evt_payment_format_last" in features.columns
    assert "profile_region" in features.columns
    assert "evaluation_time" not in features.columns


def test_run_transxion_binary_benchmark_writes_report_without_running_real_tree_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenized_dir = tmp_path / "tokenized"
    split_dir = tmp_path / "splits"
    tokenizer_dir = tmp_path / "tokenizer"
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "benchmark"
    ckpt_path = tmp_path / "best.ckpt"

    write_tokenizer(tokenizer_dir)
    write_tokenized_dataset(tokenized_dir)
    write_splits(split_dir)
    write_checkpoint(ckpt_path, tokenizer_dir)
    write_processed_transxion_like_dir(processed_dir)

    def _fake_tree_runner(train_df, valid_df, test_df, seed, output_dir):  # type: ignore[no-untyped-def]
        del train_df, valid_df, test_df, seed
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "model_name": "fake_tree",
            "best_params": {"depth": 4},
            "valid_metrics": {"pr_auc": 0.5, "roc_auc": 0.5, "f1": 0.5, "f0_5": 0.5, "threshold": 0.5, "n": 1, "pos": 1},
            "test_metrics": {"pr_auc": 0.5, "roc_auc": 0.5, "f1": 0.5, "f0_5": 0.5, "threshold": 0.5, "n": 1, "pos": 1},
        }

    monkeypatch.setattr("src.benchmarks.transxion_binary_benchmark.run_xgboost_grid_search", _fake_tree_runner)
    monkeypatch.setattr("src.benchmarks.transxion_binary_benchmark.run_catboost_grid_search", _fake_tree_runner)

    report = run_transxion_binary_benchmark(
        checkpoint=ckpt_path,
        tokenized_dir=tokenized_dir,
        processed_dir=processed_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        sample_size=6,
        batch_size=2,
        device="cpu",
        seed=7,
        repr_type="concat",
    )

    saved_report = json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["repr_type"] == "concat"
    assert set(saved_report["results"].keys()) == {"pragma_lite_logreg", "xgboost", "catboost"}
    assert (output_dir / "sampled_splits" / "train_ids.txt").exists()
    assert (output_dir / "pragma_lite_concat_train_embeddings.parquet").exists()
