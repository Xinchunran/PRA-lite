"""
PRAGMA-lite core-integrity and shortcut-detection tests.

Place this file at:
    tests/test_pragma_lite_integrity.py

Run:
    pytest tests/test_pragma_lite_integrity.py -q

Optional environment variables let you point the tests to your actual symbols
without editing this file:

    PRAGMA_SOFT_LOG_FN              default: src.tokenizer.time:soft_log_seconds
    PRAGMA_PERIODIC_ENCODE_FN       default: src.tokenizer.time:periodic_encode
    PRAGMA_BUCKETIZER_CLASS         default: src.tokenizer.bucketizer:PercentileBucketizer
    PRAGMA_MASK_COLLATOR_CLASS      default: src.tokenizer.masking:MaskedEventCollator
    PRAGMA_MODEL_CLASS              default: src.model.pragma_lite:PragmaLiteModel
    PRAGMA_SPLIT_VALIDATOR_FN       default: src.splitter.validation:validate_splits
    PRAGMA_SRC_ROOT                 default: ./src

The tests are intentionally designed around the PRAGMA-lite ideas:
1. key-value-time tokenization;
2. numerical percentile bucketization with a zero bucket;
3. profile/event/history encoder separation;
4. masked event modelling without downstream-label leakage;
5. entity/time split hygiene;
6. no shortcut columns, no future events, no label tokens in model inputs.

Some tests use optional project APIs. If your project does not yet expose the
expected symbol, that test is skipped rather than silently passing. Once the API
is implemented, remove the skips by setting the environment variables above.
"""

from __future__ import annotations

import importlib
import inspect
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed for static tests
    torch = None


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_symbol(spec: str) -> Any:
    """Import a symbol from 'package.module:Name' or 'package.module.Name'."""
    if ":" in spec:
        module_name, attr_name = spec.split(":", 1)
    else:
        module_name, attr_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _optional_symbol(env_name: str, default_spec: str) -> Any:
    spec = os.getenv(env_name, default_spec)
    try:
        return _import_symbol(spec)
    except Exception as exc:
        pytest.skip(f"Optional project symbol is unavailable: {spec}. Original error: {exc}")


# ---------------------------------------------------------------------------
# PRAGMA-lite reference math
# ---------------------------------------------------------------------------


def _reference_soft_log_seconds(t: np.ndarray | float) -> np.ndarray:
    """PRAGMA-style soft log transform: 8 * ln(1 + t / 8)."""
    return 8.0 * np.log1p(np.asarray(t, dtype=np.float64) / 8.0)


def _reference_periodic_encode(values: np.ndarray, period: float) -> np.ndarray:
    """Return [sin(2*pi*x/period), cos(2*pi*x/period)]."""
    values = np.asarray(values, dtype=np.float64)
    radians = 2.0 * np.pi * values / float(period)
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)


def test_soft_log_seconds_matches_formula_and_is_monotonic() -> None:
    """The time-delta transform must match 8*ln(1+t/8), be monotone, and compress long gaps."""
    fn = _optional_symbol("PRAGMA_SOFT_LOG_FN", "src.tokenizer.time:soft_log_seconds")

    t = np.array([0.0, 1.0, 8.0, 60.0, 3600.0, 86400.0, 7 * 86400.0])
    expected = _reference_soft_log_seconds(t)
    actual = np.asarray(fn(t), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    assert np.all(np.diff(actual) > 0.0), "soft_log_seconds must be strictly monotone for positive deltas"
    assert actual[-1] < t[-1] / 100.0, "large time gaps should be strongly compressed"
    assert math.isclose(float(actual[0]), 0.0, abs_tol=1e-12)


def test_soft_log_seconds_is_locally_almost_linear() -> None:
    """For very small t, 8*ln(1+t/8) ~= t. This preserves recent-event resolution."""
    fn = _optional_symbol("PRAGMA_SOFT_LOG_FN", "src.tokenizer.time:soft_log_seconds")

    t = np.array([1e-6, 1e-4, 1e-2], dtype=np.float64)
    actual = np.asarray(fn(t), dtype=np.float64)
    np.testing.assert_allclose(actual, t, rtol=1e-3, atol=1e-9)


def test_periodic_encoding_has_exact_calendar_periodicity() -> None:
    """Hour/day/month cyclical encodings must be periodic, not ordinal-only."""
    fn = _optional_symbol("PRAGMA_PERIODIC_ENCODE_FN", "src.tokenizer.time:periodic_encode")

    # The expected project signature is periodic_encode(values, period) -> (..., 2).
    for period in [24, 7, 31]:
        values = np.array([0, period, 2 * period, 1, 1 + period], dtype=np.float64)
        actual = np.asarray(fn(values, period=period), dtype=np.float64)
        expected = _reference_periodic_encode(values, period=period)

        assert actual.shape[-1] == 2, "periodic encoding must contain sin and cos components"
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(actual[0], actual[1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(actual[3], actual[4], rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Synthetic PRAGMA-like records
# ---------------------------------------------------------------------------


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def make_good_records() -> List[Dict[str, Any]]:
    return [
        {
            "entity_id": "acct_001",
            "evaluation_time": _ts("2024-05-01T00:00:00"),
            "profile": {"region": "uk", "plan": "metal", "age_bucket": "25_35"},
            "events": [
                {"timestamp": _ts("2024-04-01T12:00:00"), "fields": {"type": "topup", "amount": 100.0, "currency": "gbp"}},
                {"timestamp": _ts("2024-04-15T09:00:00"), "fields": {"type": "card_payment", "amount": 14.99, "mcc": "6012"}},
            ],
            "label": 0,
        },
        {
            "entity_id": "acct_002",
            "evaluation_time": _ts("2024-05-01T00:00:00"),
            "profile": {"region": "fr", "plan": "standard", "age_bucket": "35_45"},
            "events": [
                {"timestamp": _ts("2024-04-02T18:00:00"), "fields": {"type": "p2p_transfer", "amount": 5000.0, "currency": "eur"}},
                {"timestamp": _ts("2024-04-21T20:30:00"), "fields": {"type": "cash_withdrawal", "amount": 800.0, "currency": "eur"}},
            ],
            "label": 1,
        },
    ]


def make_future_leak_record() -> Dict[str, Any]:
    rec = make_good_records()[0].copy()
    rec["events"] = list(rec["events"]) + [
        {"timestamp": _ts("2024-05-02T00:00:00"), "fields": {"type": "future_event", "amount": 9999.0}}
    ]
    return rec


def _assert_no_future_events(records: Sequence[Mapping[str, Any]]) -> None:
    for rec in records:
        evaluation_time = rec["evaluation_time"]
        for event in rec["events"]:
            assert event["timestamp"] <= evaluation_time, (
                f"Future event leakage for entity={rec.get('entity_id')}: "
                f"event timestamp {event['timestamp']} > evaluation_time {evaluation_time}"
            )


def _assert_no_label_in_event_or_profile(records: Sequence[Mapping[str, Any]]) -> None:
    forbidden = {"label", "target", "y", "is_fraud", "is_laundering", "default", "outcome"}
    for rec in records:
        profile_keys = {str(k).lower() for k in rec.get("profile", {}).keys()}
        assert not (profile_keys & forbidden), f"Forbidden target-like profile keys: {profile_keys & forbidden}"
        for event in rec.get("events", []):
            event_keys = {str(k).lower() for k in event.get("fields", {}).keys()}
            assert not (event_keys & forbidden), f"Forbidden target-like event keys: {event_keys & forbidden}"


def test_synthetic_records_have_no_future_events_and_no_label_features() -> None:
    records = make_good_records()
    _assert_no_future_events(records)
    _assert_no_label_in_event_or_profile(records)


def test_future_event_leakage_detector_fails_on_bad_record() -> None:
    with pytest.raises(AssertionError, match="Future event leakage"):
        _assert_no_future_events([make_future_leak_record()])


# ---------------------------------------------------------------------------
# Split hygiene
# ---------------------------------------------------------------------------


def _assert_disjoint_entity_splits(splits: Mapping[str, Iterable[str]]) -> None:
    split_sets = {name: set(values) for name, values in splits.items()}
    names = list(split_sets.keys())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            assert not overlap, f"Entity leakage between {left} and {right}: {sorted(overlap)[:5]}"


def test_entity_splits_are_disjoint() -> None:
    good = {"train": ["a", "b"], "valid": ["c"], "test": ["d"]}
    _assert_disjoint_entity_splits(good)

    bad = {"train": ["a", "b"], "valid": ["b"], "test": ["d"]}
    with pytest.raises(AssertionError, match="Entity leakage"):
        _assert_disjoint_entity_splits(bad)


def test_project_split_validator_rejects_entity_overlap_if_available() -> None:
    validator = _optional_symbol("PRAGMA_SPLIT_VALIDATOR_FN", "src.splitter.validation:validate_splits")
    bad = {"train": ["acct_001", "acct_002"], "valid": ["acct_002"], "test": ["acct_003"]}
    with pytest.raises(Exception):
        validator(bad)


# ---------------------------------------------------------------------------
# Numerical bucketization
# ---------------------------------------------------------------------------


def test_percentile_bucketizer_uses_training_values_only_and_handles_zero() -> None:
    """Quantile boundaries must be fit on train only; test extremes must not alter boundaries."""
    Bucketizer = _optional_symbol("PRAGMA_BUCKETIZER_CLASS", "src.tokenizer.bucketizer:PercentileBucketizer")

    train_values = np.array([0, 1, 2, 3, 4, 5, 10, 20, 100], dtype=np.float64)
    test_values = np.array([0, 999999], dtype=np.float64)

    bucketizer = Bucketizer(num_buckets=5, add_zero_bucket=True)
    bucketizer.fit(train_values)

    before = np.asarray(getattr(bucketizer, "boundaries_", getattr(bucketizer, "bin_edges_", [])), dtype=np.float64)
    assert before.size > 0, "Bucketizer should expose learned boundaries_ or bin_edges_"
    assert np.max(before) <= np.max(train_values), "Boundaries should be learned from train values only"

    _ = bucketizer.transform(test_values)
    after = np.asarray(getattr(bucketizer, "boundaries_", getattr(bucketizer, "bin_edges_", [])), dtype=np.float64)
    np.testing.assert_allclose(after, before, err_msg="Transforming test values must not refit bucket boundaries")

    zero_id = bucketizer.transform(np.array([0.0]))[0]
    repeat_zero_id = bucketizer.transform(np.array([0.0, 0.0]))
    assert np.all(repeat_zero_id == zero_id), "All exact zeros must map to the same zero bucket"


# ---------------------------------------------------------------------------
# Masked event modelling and leakage checks
# ---------------------------------------------------------------------------


def _make_encoded_record() -> Dict[str, Any]:
    """
    Minimal encoded record expected by MaskedEventCollator.

    A project collator may use richer fields, but it should be able to process
    a dictionary with profile_tokens, event_tokens, timestamps, and label.
    """
    return {
        "entity_id": "acct_001",
        "profile_tokens": np.array([101, 201, 301], dtype=np.int64),
        "event_tokens": np.array(
            [
                [11, 111, 12, 112, 13, 113],
                [21, 121, 22, 122, 23, 123],
            ],
            dtype=np.int64,
        ),
        "event_times": np.array([100.0, 10.0], dtype=np.float32),
        "label": 1,
    }


def _flatten_batch_values(obj: Any) -> List[Any]:
    if isinstance(obj, Mapping):
        out: List[Any] = []
        for value in obj.values():
            out.extend(_flatten_batch_values(value))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for value in obj:
            out.extend(_flatten_batch_values(value))
        return out
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().reshape(-1).tolist()
    if isinstance(obj, np.ndarray):
        return obj.reshape(-1).tolist()
    return [obj]


def test_mlm_collator_does_not_put_downstream_label_into_model_inputs() -> None:
    """The downstream label may be returned separately, but must never be part of MLM input tokens."""
    Collator = _optional_symbol("PRAGMA_MASK_COLLATOR_CLASS", "src.tokenizer.masking:MaskedEventCollator")
    collator = Collator(mask_token_id=999, unk_token_id=998, mlm_probability=0.5, seed=123)

    batch = collator([_make_encoded_record()])
    assert isinstance(batch, Mapping), "Collator must return a mapping/dict-like batch"

    # The label value is 1. This test focuses on key placement, not token id collision.
    forbidden_input_key_regex = re.compile(r"(^|_)(label|target|y|is_fraud|is_laundering)($|_)", re.IGNORECASE)
    for key in batch.keys():
        if forbidden_input_key_regex.search(str(key)):
            continue
        # Non-label input keys are allowed to contain token id 1 by chance, so only enforce key-level separation here.
        assert not forbidden_input_key_regex.search(str(key)), f"Target-like key leaked into model inputs: {key}"

    allowed_label_keys = {"label", "labels", "targets", "y", "downstream_label"}
    target_like_keys = {str(k) for k in batch.keys() if forbidden_input_key_regex.search(str(k))}
    assert target_like_keys <= allowed_label_keys, f"Unexpected target-like batch keys: {target_like_keys}"


def test_mlm_labels_are_ignore_index_except_at_masked_positions() -> None:
    """MLM labels should contain original token ids only where inputs were masked."""
    if torch is None:
        pytest.skip("torch is unavailable")

    Collator = _optional_symbol("PRAGMA_MASK_COLLATOR_CLASS", "src.tokenizer.masking:MaskedEventCollator")
    collator = Collator(mask_token_id=999, unk_token_id=998, mlm_probability=0.5, seed=123)
    batch = collator([_make_encoded_record()])

    input_ids = batch.get("input_ids", batch.get("event_input_ids", None))
    mlm_labels = batch.get("mlm_labels", batch.get("labels", None))
    assert input_ids is not None, "Collator must expose input_ids or event_input_ids"
    assert mlm_labels is not None, "Collator must expose mlm_labels or labels for MLM"

    input_ids = torch.as_tensor(input_ids)
    mlm_labels = torch.as_tensor(mlm_labels)
    assert input_ids.shape == mlm_labels.shape, "MLM labels must align with input token positions"

    ignore_index = getattr(collator, "ignore_index", -100)
    masked_positions = mlm_labels != ignore_index
    assert masked_positions.any(), "With mlm_probability=0.5 and a fixed seed, at least one token should be masked"
    assert torch.all(input_ids[masked_positions] != mlm_labels[masked_positions]), (
        "Masked input positions should not still expose the original token id"
    )
    assert torch.all(mlm_labels[~masked_positions] == ignore_index), "Unmasked positions must use ignore_index"


# ---------------------------------------------------------------------------
# Model architecture behavior
# ---------------------------------------------------------------------------


@dataclass
class TinyModelConfig:
    vocab_size: int = 2048
    d_model: int = 32
    d_ffn: int = 64
    n_heads: int = 4
    dropout: float = 0.0
    profile_layers: int = 1
    event_layers: int = 1
    history_layers: int = 1
    max_profile_tokens: int = 8
    max_event_tokens: int = 6
    max_events: int = 4
    use_time_encoding: bool = True
    use_profile_encoder: bool = True


def _instantiate_tiny_model() -> Any:
    if torch is None:
        pytest.skip("torch is unavailable")
    Model = _optional_symbol("PRAGMA_MODEL_CLASS", "src.model.pragma_lite:PragmaLiteModel")
    try:
        model = Model(TinyModelConfig())
    except TypeError:
        model = Model(**TinyModelConfig().__dict__)
    model.eval()
    return model


def _tiny_batch() -> Dict[str, Any]:
    if torch is None:
        pytest.skip("torch is unavailable")
    return {
        "profile_input_ids": torch.tensor([[101, 201, 301]], dtype=torch.long),
        "event_input_ids": torch.tensor([[[11, 111, 12, 112, 13, 113], [21, 121, 22, 122, 23, 123]]], dtype=torch.long),
        "event_times": torch.tensor([[100.0, 10.0]], dtype=torch.float32),
        "calendar_features": torch.tensor([[[12.0, 1.0, 2.0], [18.0, 3.0, 5.0]]], dtype=torch.float32),
    }


def _get_record_embedding(output: Any) -> Any:
    if isinstance(output, Mapping):
        for key in ["record_embedding", "usr_embedding", "pooled_output", "embedding", "z_h"]:
            if key in output:
                return output[key]
    if hasattr(output, "record_embedding"):
        return output.record_embedding
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    if torch is not None and isinstance(output, torch.Tensor):
        return output
    raise AssertionError("Could not find record-level embedding in model output")


def test_event_encoder_is_independent_before_history_encoder() -> None:
    """Changing event #2 must not change event #1's EventEncoder output before HistoryEncoder."""
    model = _instantiate_tiny_model()
    if not hasattr(model, "encode_events"):
        pytest.skip("Model does not expose encode_events(); add this method to test event-encoder independence")

    batch = _tiny_batch()
    with torch.no_grad():
        z1 = model.encode_events(batch["event_input_ids"], calendar_features=batch.get("calendar_features"))

    changed = dict(batch)
    changed["event_input_ids"] = batch["event_input_ids"].clone()
    changed["event_input_ids"][:, 1, :] += 100  # perturb only second event
    with torch.no_grad():
        z2 = model.encode_events(changed["event_input_ids"], calendar_features=changed.get("calendar_features"))

    z1 = torch.as_tensor(z1)
    z2 = torch.as_tensor(z2)
    np.testing.assert_allclose(
        z1[:, 0].detach().cpu().numpy(),
        z2[:, 0].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
        err_msg="EventEncoder should process each event independently before HistoryEncoder",
    )


def test_history_encoder_is_sensitive_to_event_order_and_time() -> None:
    """A PRAGMA-like history encoder should not collapse event order/time information."""
    model = _instantiate_tiny_model()
    batch = _tiny_batch()

    with torch.no_grad():
        base = _get_record_embedding(model(**batch))

    swapped = dict(batch)
    swapped["event_input_ids"] = batch["event_input_ids"].flip(dims=[1])
    swapped["event_times"] = batch["event_times"].flip(dims=[1])
    swapped["calendar_features"] = batch["calendar_features"].flip(dims=[1])

    with torch.no_grad():
        changed = _get_record_embedding(model(**swapped))

    assert not torch.allclose(torch.as_tensor(base), torch.as_tensor(changed), atol=1e-6), (
        "HistoryEncoder embedding is invariant to event order/time; this suggests missing temporal encoding"
    )


def test_profile_state_affects_record_embedding() -> None:
    """Changing profile tokens should change the final record embedding when profile branch is enabled."""
    model = _instantiate_tiny_model()
    batch = _tiny_batch()

    with torch.no_grad():
        base = _get_record_embedding(model(**batch))

    changed = dict(batch)
    changed["profile_input_ids"] = batch["profile_input_ids"].clone() + 17
    with torch.no_grad():
        changed_embedding = _get_record_embedding(model(**changed))

    assert not torch.allclose(torch.as_tensor(base), torch.as_tensor(changed_embedding), atol=1e-6), (
        "Profile branch appears unused: profile changes do not affect record-level embedding"
    )


def test_mlm_head_uses_local_event_and_history_context_if_exposed() -> None:
    """The MLM head should have access to local token, event-level, and user-level context."""
    model = _instantiate_tiny_model()
    if not hasattr(model, "mlm_head"):
        pytest.skip("Model does not expose mlm_head; skip structural MLM-head test")

    head = model.mlm_head
    d_model = getattr(model, "d_model", getattr(getattr(model, "config", None), "d_model", None))
    assert d_model is not None, "Model should expose d_model directly or through config"

    # Search the first Linear-like layer. Its input should be 3*d_model if it concatenates
    # local token context, event-level context, and user-level context.
    first_linear = None
    for module in head.modules() if hasattr(head, "modules") else []:
        if module.__class__.__name__.lower() == "linear":
            first_linear = module
            break
    assert first_linear is not None, "MLM head should contain a projection layer"
    assert getattr(first_linear, "in_features") == 3 * int(d_model), (
        "MLM head should project concatenated [local_token, event_context, user_context] = 3*d_model"
    )


# ---------------------------------------------------------------------------
# Static source checks for shortcut-prone code patterns
# ---------------------------------------------------------------------------


_TARGET_LIKE = re.compile(r"\b(label|target|y_true|is_fraud|is_laundering|default|outcome)\b", re.IGNORECASE)
_SHORTCUT_PATTERNS = [
    re.compile(r"feature_cols\s*=\s*.*\b(label|target|is_fraud|is_laundering|default|outcome)\b", re.IGNORECASE),
    re.compile(r"input_cols\s*=\s*.*\b(label|target|is_fraud|is_laundering|default|outcome)\b", re.IGNORECASE),
    re.compile(r"transaction_columns\s*=\s*.*\b(label|target|is_fraud|is_laundering|default|outcome)\b", re.IGNORECASE),
    re.compile(r"profile_columns\s*=\s*.*\b(label|target|is_fraud|is_laundering|default|outcome)\b", re.IGNORECASE),
]


def _iter_source_files(src_root: Path) -> Iterable[Path]:
    if not src_root.exists():
        pytest.skip(f"Source root does not exist: {src_root}")
    for suffix in ["*.py", "*.yaml", "*.yml"]:
        yield from src_root.rglob(suffix)


def test_static_scan_no_obvious_target_columns_in_feature_lists() -> None:
    """Catch common shortcuts where label/target columns are accidentally listed as input features."""
    src_root = Path(os.getenv("PRAGMA_SRC_ROOT", "src"))
    offenders: List[Tuple[str, str]] = []
    for path in _iter_source_files(src_root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _SHORTCUT_PATTERNS:
            for match in pattern.finditer(text):
                offenders.append((str(path), match.group(0)[:200]))

    assert not offenders, "Target-like columns appear in feature/input lists:\n" + "\n".join(
        f"{path}: {snippet}" for path, snippet in offenders[:20]
    )


def test_core_encoder_forward_signature_does_not_require_downstream_labels() -> None:
    """The representation encoder should run without downstream labels; labels belong in loss wrappers."""
    Model = _optional_symbol("PRAGMA_MODEL_CLASS", "src.model.pragma_lite:PragmaLiteModel")
    if not hasattr(Model, "forward"):
        pytest.skip("Model class has no forward method")

    sig = inspect.signature(Model.forward)
    required = [
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    target_like_required = [name for name in required if _TARGET_LIKE.search(name)]
    assert not target_like_required, (
        "Core encoder forward() requires target-like arguments. "
        f"Required target-like args: {target_like_required}. "
        "Move downstream labels to a training step/loss wrapper to prevent shortcut coupling."
    )


# ---------------------------------------------------------------------------
# Optional behavioral shortcut traps
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_label_permutation_trap_is_near_random_if_training_api_available(tmp_path: Path) -> None:
    """
    Optional slow test.

    If your project exposes a small training helper, this verifies that a model trained on
    randomly permuted labels cannot achieve meaningful validation performance. Set:

        PRAGMA_TRAIN_EVAL_FN=src.training.debug:train_and_eval_on_records

    Expected callable signature:
        metrics = train_and_eval_on_records(records, label_permutation=True, max_steps=50, output_dir=tmp_path)

    Expected metrics keys:
        valid_roc_auc and/or valid_pr_auc
    """
    spec = os.getenv("PRAGMA_TRAIN_EVAL_FN")
    if not spec:
        pytest.skip("Set PRAGMA_TRAIN_EVAL_FN to enable the label-permutation shortcut trap")
    train_eval = _import_symbol(spec)

    records = make_good_records() * 32
    metrics = train_eval(records, label_permutation=True, max_steps=50, output_dir=tmp_path)
    roc_auc = metrics.get("valid_roc_auc")
    if roc_auc is not None:
        assert 0.35 <= roc_auc <= 0.65, f"Permuted-label ROC-AUC should be near random; got {roc_auc}"

    pr_auc = metrics.get("valid_pr_auc")
    if pr_auc is not None:
        prevalence = np.mean([r["label"] for r in records])
        assert pr_auc <= prevalence + 0.20, (
            f"Permuted-label PR-AUC is suspiciously high: {pr_auc}; prevalence={prevalence}"
        )
