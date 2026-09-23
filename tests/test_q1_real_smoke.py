"""Opt-in smoke coverage for the configured real Question 1 environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from e_mosei_audit.q1.config import load_config
from e_mosei_audit.q1.runner import run_q1


@pytest.mark.integration
def test_real_q1_smoke(tmp_path: Path) -> None:
    config_path = os.environ.get("E_MOSEI_Q1_CONFIG")
    if not config_path:
        pytest.skip("set E_MOSEI_Q1_CONFIG to run the real Q1 smoke test")

    config = load_config(Path(config_path))
    config = config.with_output_dir(tmp_path / "q1-smoke")

    summary = run_q1(config, limit=1)

    assert summary["coverage_count"] == 1
    assert summary["success_count"] == 1
    assert summary["failed_count"] == 0
