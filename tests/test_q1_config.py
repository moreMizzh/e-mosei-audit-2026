from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from e_mosei_audit.q1.config import Q1Config, load_config


def _write_valid_config(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"archive")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"tool")
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    output_dir = tmp_path / "output" / "q1"
    output_dir.parent.mkdir()

    config_path = tmp_path / "q1.toml"
    config_path.write_text(
        """[paths]
audit_dir = "audit"
archive = "data.zip"
seven_zip = "7za"
ffmpeg = "ffmpeg"
model_cache = "models"
output_dir = "output/q1"
"""
    )

    return config_path, {
        "audit_dir": audit_dir,
        "archive": archive,
        "seven_zip": seven_zip,
        "ffmpeg": ffmpeg,
        "model_cache": model_cache,
        "output_dir": output_dir,
    }


def test_load_config_rejects_missing_audit_dir(tmp_path: Path) -> None:
    config_path, _ = _write_valid_config(tmp_path)
    config_path.write_text(
        """[paths]
archive = "data.zip"
seven_zip = "7za"
ffmpeg = "ffmpeg"
model_cache = "models"
output_dir = "output/q1"
"""
    )

    with pytest.raises(ValueError, match="audit_dir"):
        load_config(config_path)


def test_load_config_resolves_all_required_paths(tmp_path: Path) -> None:
    config_path, expected = _write_valid_config(tmp_path)

    config = load_config(config_path)

    assert isinstance(config, Q1Config)
    assert config.audit_dir == expected["audit_dir"]
    assert config.archive == expected["archive"]
    assert config.seven_zip == expected["seven_zip"]
    assert config.ffmpeg == expected["ffmpeg"]
    assert config.model_cache == expected["model_cache"]
    assert config.output_dir == expected["output_dir"]


def test_loaded_config_is_frozen(tmp_path: Path) -> None:
    config_path, _ = _write_valid_config(tmp_path)
    config = load_config(config_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.audit_dir = tmp_path


def test_load_config_rejects_missing_paths_table(tmp_path: Path) -> None:
    config_path = tmp_path / "q1.toml"
    config_path.write_text("")

    with pytest.raises(ValueError, match="paths"):
        load_config(config_path)


@pytest.mark.parametrize(
    "replacement",
    ["archive = 42", 'archive = ""'],
    ids=["non-string", "empty"],
)
def test_load_config_rejects_invalid_path_values(tmp_path: Path, replacement: str) -> None:
    config_path, _ = _write_valid_config(tmp_path)
    config_path.write_text(
        config_path.read_text().replace('archive = "data.zip"', replacement)
    )

    with pytest.raises(ValueError, match="archive"):
        load_config(config_path)


def test_load_config_rejects_unexpected_path_field(tmp_path: Path) -> None:
    config_path, _ = _write_valid_config(tmp_path)
    config_path.write_text(f"{config_path.read_text()}unexpected = \"value\"\n")

    with pytest.raises(ValueError, match="unexpected"):
        load_config(config_path)


def test_load_config_rejects_directory_for_required_file(tmp_path: Path) -> None:
    config_path, expected = _write_valid_config(tmp_path)
    expected["archive"].unlink()
    expected["archive"].mkdir()

    with pytest.raises(ValueError, match="archive"):
        load_config(config_path)


def test_load_config_reports_missing_output_parent(tmp_path: Path) -> None:
    config_path, expected = _write_valid_config(tmp_path)
    expected["output_dir"].parent.rmdir()

    with pytest.raises(
        ValueError, match="output_dir parent must be an existing directory"
    ):
        load_config(config_path)


def test_load_config_rejects_missing_output_dir_without_key_error(tmp_path: Path) -> None:
    config_path, _ = _write_valid_config(tmp_path)
    config_path.write_text(
        """[paths]
audit_dir = "audit"
archive = "data.zip"
seven_zip = "7za"
ffmpeg = "ffmpeg"
model_cache = "models"
"""
    )

    with pytest.raises(ValueError, match="output_dir"):
        load_config(config_path)
