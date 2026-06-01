from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarks.ibm_aml_downstream_benchmark import _plot_test_metrics, _run_model_suite


def test_plot_test_metrics_supports_single_available_model(tmp_path: Path) -> None:
    report = {
        "results": {
            "pragma_lite_logreg": {
                "test_metrics": {
                    "pr_auc": 0.12,
                    "roc_auc": 0.34,
                    "f1": 0.56,
                    "f0_5": 0.78,
                }
            }
        }
    }

    _plot_test_metrics(report, plots_dir=tmp_path)

    assert (tmp_path / "test_metric_bars.png").exists()


def test_run_model_suite_skips_optional_missing_dependencies(tmp_path: Path, monkeypatch) -> None:
    raw_df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "label": [0, 1],
            "split": ["train", "test"],
        }
    )
    embeddings_df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "label": [0, 1],
            "split": ["train", "test"],
            "embedding_0": [0.0, 1.0],
        }
    )
    metrics_dir = tmp_path / "metrics"
    predictions_dir = tmp_path / "predictions"
    metrics_dir.mkdir()
    predictions_dir.mkdir()

    def _fake_logreg(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return (
            {
                "model_name": "pragma_lite_logreg",
                "best_params": {"C": 1.0},
                "cv_metrics": {"pr_auc": 0.5},
                "test_metrics": {"pr_auc": 0.4, "roc_auc": 0.6, "f1": 0.3, "f0_5": 0.2},
            },
            [{"model_name": "pragma_lite_logreg", "cv_pr_auc": 0.5}],
        )

    def _missing_xgboost(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise ImportError("xgboost is required for the IBM AML downstream benchmark")

    def _missing_catboost(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise ImportError("catboost is required for the IBM AML downstream benchmark")

    monkeypatch.setattr("src.benchmarks.ibm_aml_downstream_benchmark.run_pragma_lite_cv", _fake_logreg)
    monkeypatch.setattr("src.benchmarks.ibm_aml_downstream_benchmark.run_xgboost_cv", _missing_xgboost)
    monkeypatch.setattr("src.benchmarks.ibm_aml_downstream_benchmark.run_catboost_cv", _missing_catboost)

    results, cv_rows, skipped_models = _run_model_suite(
        raw_df=raw_df,
        embeddings_df=embeddings_df,
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
        seed=7,
        cv_folds=2,
    )

    assert set(results.keys()) == {"pragma_lite_logreg"}
    assert set(cv_rows.keys()) == {"pragma_lite_logreg"}
    assert set(skipped_models.keys()) == {"xgboost", "catboost"}
    assert skipped_models["xgboost"]["status"] == "skipped_missing_dependency"
    assert skipped_models["catboost"]["status"] == "skipped_missing_dependency"
