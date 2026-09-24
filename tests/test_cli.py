from __future__ import annotations

from pathlib import Path

from e_mosei_audit import cli
from e_mosei_audit.q1.config import Q1Config
from e_mosei_audit.q2.config import Q2Config


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


def _q2_config(tmp_path: Path) -> Q2Config:
    return Q2Config(
        archive=tmp_path / "data.zip",
        seven_zip=tmp_path / "7za",
        bert_model=tmp_path / "bert",
        output_dir=tmp_path / "q2-output",
        seed=7,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        weight_decay=0.0,
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        regression_loss_weight=0.5,
        class_weight_exponent=1.0,
        synthetic_missingness_enabled=True,
        device="cpu",
    )


def test_train_q2_loads_config_and_reports_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    config = _q2_config(tmp_path)
    observed = {}
    monkeypatch.setattr(cli, "load_q2_config", lambda path: config)

    def fake_run_q2(q2_config):
        observed["config"] = q2_config
        return {"best_epoch": 0, "attachment3_count": 30, "accuracy": 0.5}

    monkeypatch.setattr(cli, "run_q2", fake_run_q2)

    status = cli.main(["train-q2", "--config", str(tmp_path / "q2.toml")])

    assert status == 0
    assert observed["config"] == config
    assert '"attachment3_count": 30' in capsys.readouterr().out


def test_train_q2_check_runs_only_preflight(monkeypatch, capsys, tmp_path: Path) -> None:
    config = _q2_config(tmp_path)
    monkeypatch.setattr(cli, "load_q2_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "check_q2",
        lambda checked_config: {"train_count": 3, "valid_count": 2, "attachment3_count": 30},
    )
    monkeypatch.setattr(cli, "run_q2", lambda *args: (_ for _ in ()).throw(AssertionError("must not train")))

    status = cli.main(["train-q2", "--config", str(tmp_path / "q2.toml"), "--check"])

    assert status == 0
    assert '"attachment3_count": 30' in capsys.readouterr().out


def test_evaluate_q2_valid_forwards_saved_run_and_new_output(monkeypatch, capsys, tmp_path: Path) -> None:
    observed = {}

    def fake_evaluate(run_dir, output_path):
        observed["run_dir"] = run_dir
        observed["output_path"] = output_path
        return {"sample_count": 728, "metrics": {"macro_f1": 0.6}}

    monkeypatch.setattr(cli, "evaluate_saved_q2_valid", fake_evaluate)

    status = cli.main(
        [
            "evaluate-q2-valid",
            "--run-dir",
            str(tmp_path / "saved-run"),
            "--output",
            str(tmp_path / "valid-report.json"),
        ]
    )

    assert status == 0
    assert observed == {"run_dir": tmp_path / "saved-run", "output_path": tmp_path / "valid-report.json"}
    assert '"sample_count": 728' in capsys.readouterr().out
