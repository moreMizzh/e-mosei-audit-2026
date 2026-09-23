from __future__ import annotations

import os
from pathlib import Path

import pytest

from e_mosei_audit.archive import SevenZipArchive
from e_mosei_audit.workflow import run_audit


@pytest.mark.integration
def test_full_competition_archive_contract(tmp_path) -> None:
    archive_value = os.environ.get("E_MOSEI_ARCHIVE")
    executable_value = os.environ.get("E_MOSEI_7ZA")
    if not archive_value or not executable_value:
        pytest.skip("set E_MOSEI_ARCHIVE and E_MOSEI_7ZA to enable the real-archive audit")

    archive_path = Path(archive_value)
    summary = run_audit(
        SevenZipArchive(archive_path, Path(executable_value)),
        tmp_path / "audit",
        archive_name=archive_path.name,
    )

    assert summary["member_count"] == 294
    assert summary["raw_error_count"] == 0
    assert summary["special_record_count"] == 100
    assert summary["special_error_count"] == 0
