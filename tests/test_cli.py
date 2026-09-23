from __future__ import annotations

from pathlib import Path

from e_mosei_audit import cli
from e_mosei_audit.q1.config import Q1Config


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


def _q1_config(tmp_path: Path) -> Q1Config:
    return Q1Config(
        audit_dir=tmp_path / "audit",
        archive=tmp_path / "data.zip",
        seven_zip=tmp_path / "7za",
        ffmpeg=tmp_path / "ffmpeg",
        model_cache=tmp_path / "models",
        output_dir=tmp_path / "configured-output",
    )


def test_extract_q1_loads_config_overrides_output_and_forwards_limit(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    config = _q1_config(tmp_path)
    observed = {}

    monkeypatch.setattr(cli, "load_config", lambda path: config)

    def fake_run_q1(q1_config, *, limit):
        observed["config"] = q1_config
        observed["limit"] = limit
        return {"coverage_count": 2, "success_count": 2, "failed_count": 0}

    monkeypatch.setattr(cli, "run_q1", fake_run_q1)

    status = cli.main(
        [
            "extract-q1",
            "--config",
            str(tmp_path / "q1.toml"),
            "--output",
            str(tmp_path / "cli-output"),
            "--limit",
            "2",
        ]
    )

    assert status == 0
    assert observed["config"].output_dir == tmp_path / "cli-output"
    assert observed["limit"] == 2
    assert '"failed_count": 0' in capsys.readouterr().out


def test_extract_q1_returns_one_when_coverage_contains_failed_samples(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: _q1_config(tmp_path))
    monkeypatch.setattr(
        cli,
        "run_q1",
        lambda *args, **kwargs: {
            "coverage_count": 2,
            "success_count": 1,
            "failed_count": 1,
        },
    )

    status = cli.main(
        [
            "extract-q1",
            "--config",
            str(tmp_path / "q1.toml"),
            "--output",
            str(tmp_path / "cli-output"),
        ]
    )

    assert status == 1


def test_extract_q1_returns_two_for_config_errors(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: (_ for _ in ()).throw(ValueError("invalid config")),
    )

    status = cli.main(
        [
            "extract-q1",
            "--config",
            str(tmp_path / "q1.toml"),
            "--output",
            str(tmp_path / "cli-output"),
        ]
    )

    assert status == 2
    assert "invalid config" in capsys.readouterr().err


def test_extract_q1_returns_two_for_ffmpeg_preflight_error(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: _q1_config(tmp_path))
    monkeypatch.setattr(
        cli,
        "run_q1",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("ffmpeg preflight requires an executable regular file")
        ),
    )

    status = cli.main(
        [
            "extract-q1",
            "--config",
            str(tmp_path / "q1.toml"),
            "--output",
            str(tmp_path / "cli-output"),
        ]
    )

    assert status == 2
    assert "ffmpeg preflight" in capsys.readouterr().err
