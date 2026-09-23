from __future__ import annotations

from pathlib import Path

from e_mosei_audit import cli


def test_audit_command_passes_explicit_paths_to_workflow(monkeypatch, capsys, tmp_path) -> None:
    observed = {}

    def fake_run_audit(archive, output_dir, *, archive_name):
        observed["archive_path"] = archive.archive_path
        observed["executable"] = archive.executable
        observed["output_dir"] = output_dir
        observed["archive_name"] = archive_name
        return {"raw_error_count": 0, "special_error_count": 0}

    monkeypatch.setattr(cli, "run_audit", fake_run_audit)

    status = cli.main(
        [
            "audit",
            "--archive",
            str(tmp_path / "data.zip"),
            "--seven-zip",
            str(tmp_path / "7za"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert status == 0
    assert observed == {
        "archive_path": tmp_path / "data.zip",
        "executable": tmp_path / "7za",
        "output_dir": tmp_path / "output",
        "archive_name": "data.zip",
    }
    assert '"raw_error_count": 0' in capsys.readouterr().out


def test_audit_command_returns_nonzero_for_feature_contract_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "run_audit",
        lambda *args, **kwargs: {
            "raw_error_count": 0,
            "feature_error_count": 1,
            "special_error_count": 0,
        },
    )

    status = cli.main(
        [
            "audit",
            "--archive",
            str(tmp_path / "data.zip"),
            "--seven-zip",
            str(tmp_path / "7za"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert status == 1
