from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]) -> object:
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    strict_files = {
        "test_pragma_lite_rope_pipeline.py",
        "test_pragma_lite_backbone_contract.py",
        "test_structured_data_pipeline.py",
    }
    skipped: list[str] = []
    for item in session.items:
        if Path(str(item.fspath)).name not in strict_files:
            continue
        rep = getattr(item, "rep_call", None)
        if rep is not None and rep.skipped:
            skipped.append(item.nodeid)
    if skipped:
        session.exitstatus = 1
