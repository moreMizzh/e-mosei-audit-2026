from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_PATH_FIELDS = (
    "audit_dir",
    "archive",
    "seven_zip",
    "ffmpeg",
    "model_cache",
    "output_dir",
)


@dataclass(frozen=True)
class Q1Config:
    audit_dir: Path
    archive: Path
    seven_zip: Path
    ffmpeg: Path
    model_cache: Path
    output_dir: Path


def load_config(path: Path) -> Q1Config:
    config_path = path.resolve()
    with config_path.open("rb") as config_file:
        document = tomllib.load(config_file)

    paths = document.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("paths must be a TOML table")

    missing_fields = [field for field in _PATH_FIELDS if field not in paths]
    if missing_fields:
        raise ValueError(f"missing required path field: {missing_fields[0]}")

    unexpected_fields = sorted(set(paths) - set(_PATH_FIELDS))
    if unexpected_fields:
        raise ValueError(f"unexpected path field: {unexpected_fields[0]}")

    resolved_paths = {
        field: _resolve_path(paths[field], field, config_path.parent)
        for field in _PATH_FIELDS
    }

    _require_directory(resolved_paths["audit_dir"], "audit_dir")
    _require_file(resolved_paths["archive"], "archive")
    _require_file(resolved_paths["seven_zip"], "seven_zip")
    _require_file(resolved_paths["ffmpeg"], "ffmpeg")
    _require_directory(resolved_paths["model_cache"], "model_cache")
    _require_directory(resolved_paths["output_dir"].parent, "output_dir")

    return Q1Config(**resolved_paths)


def _resolve_path(value: object, field: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")

    entry = Path(value)
    if not entry.is_absolute():
        entry = base_dir / entry
    return entry.resolve()


def _require_directory(path: Path, field: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field} must be an existing directory: {path}")


def _require_file(path: Path, field: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field} must be an existing regular file: {path}")
