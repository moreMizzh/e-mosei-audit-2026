from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SLOT_COUNT = 50
_SLOT_BOUNDARY_ATOL = 1e-12
_FEATURE_SHAPES = {
    "text": (50, 768),
    "audio": (50, 50),
    "vision": (50, 56),
}
_COVERAGE_COLUMNS = (
    "sample_id",
    "video_id",
    "clip_id",
    "member_path",
    "duration_seconds",
    "status",
    "feature_path",
    "evidence_path",
    "text_shape",
    "audio_shape",
    "vision_shape",
    "failure_reason",
)
_FileFingerprint = tuple[int, int, int, int, bytes]


@dataclass(frozen=True)
class SampleFeatures:
    sample_id: str
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    slots: np.ndarray
    text_mask: np.ndarray
    audio_mask: np.ndarray
    vision_mask: np.ndarray

    def write(self, path: Path) -> None:
        target = _validate_feature_target(path)
        arrays = _validated_feature_arrays(self)
        _reject_duplicate_sample_id(target, self.sample_id)

        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sample_id=np.asarray(self.sample_id),
            text=arrays["text"],
            audio=arrays["audio"],
            vision=arrays["vision"],
            slots=arrays["slots"],
            text_mask=arrays["text_mask"],
            audio_mask=arrays["audio_mask"],
            vision_mask=arrays["vision_mask"],
        )


def write_coverage(
    path: Path, rows: list[Mapping[str, object]], *, expected_count: int
) -> dict[str, int]:
    output, summary = _validate_coverage_targets(path)
    normalized_rows, counts = _validate_coverage_rows(rows, expected_count)
    _write_coverage_outputs(output, summary, normalized_rows, counts)
    return counts


def _validated_feature_arrays(features: SampleFeatures) -> dict[str, np.ndarray]:
    _require_sample_id(features.sample_id)
    arrays = {
        name: _as_float16_feature(getattr(features, name), name, shape)
        for name, shape in _FEATURE_SHAPES.items()
    }
    arrays["slots"] = _as_slots(features.slots)
    arrays["text_mask"] = _as_mask(features.text_mask, "text_mask")
    arrays["audio_mask"] = _as_mask(features.audio_mask, "audio_mask")
    arrays["vision_mask"] = _as_mask(features.vision_mask, "vision_mask")
    return arrays


def _as_float16_feature(value: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must be a real numeric array")

    try:
        numeric = value.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real numeric array") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        stored = value.astype(np.float16)
    if not np.all(np.isfinite(stored)):
        raise ValueError(f"{name} must be representable as finite float16")
    return stored


def _as_slots(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (_SLOT_COUNT, 2):
        raise ValueError("slots must have shape (50, 2)")
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
        raise ValueError("slots must be a real numeric array")

    try:
        slots = value.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("slots must be a real numeric array") from error
    if not np.all(np.isfinite(slots)):
        raise ValueError("slots must be finite")
    if np.any(slots[:, 1] <= slots[:, 0]):
        raise ValueError("slots must have positive intervals")

    boundary_deltas = slots[1:, 0] - slots[:-1, 1]
    if np.any(np.abs(boundary_deltas) > _SLOT_BOUNDARY_ATOL):
        raise ValueError("slots must be contiguous")
    slots[1:, 0] = slots[:-1, 1]
    return slots


def _as_mask(value: object, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (_SLOT_COUNT,)
        or value.dtype != np.dtype(bool)
    ):
        raise ValueError(f"{name} must be a bool array with shape (50,)")
    return value


def _validate_feature_target(path: Path) -> Path:
    if not isinstance(path, Path) or path.suffix != ".npz":
        raise ValueError("feature target must be a .npz Path")
    if _path_exists(path):
        raise ValueError(f"feature target already exists: {path}")
    return path


def _require_sample_id(sample_id: object) -> None:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")


def _reject_duplicate_sample_id(target: Path, sample_id: str) -> None:
    if not target.parent.is_dir():
        return

    for candidate in target.parent.glob("*.npz"):
        try:
            with np.load(candidate, allow_pickle=False) as stored:
                existing_id = np.asarray(stored["sample_id"])
        except (KeyError, OSError, ValueError, zipfile.BadZipFile):
            continue
        if existing_id.shape == () and existing_id.item() == sample_id:
            raise ValueError(f"duplicate sample_id: {sample_id}")


def _validate_coverage_targets(path: Path) -> tuple[Path, Path]:
    if not isinstance(path, Path) or path.name != "q1_samples.csv":
        raise ValueError("coverage output must be named q1_samples.csv")
    if not path.parent.is_dir():
        raise ValueError(f"coverage output parent must exist: {path.parent}")

    summary = path.with_name("q1_summary.json")
    if _path_exists(path) or _path_exists(summary):
        raise ValueError("coverage output or summary already exists")
    return path, summary


def _validate_coverage_rows(
    rows: list[Mapping[str, object]], expected_count: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if isinstance(expected_count, (bool, np.bool_)) or not isinstance(
        expected_count, (int, np.integer)
    ) or expected_count < 0:
        raise ValueError("expected_count must be a non-negative integer")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    if len(rows) != expected_count:
        raise ValueError("coverage row count does not match expected_count")

    normalized_rows: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    success_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("coverage row must be a mapping")
        if set(row) != set(_COVERAGE_COLUMNS):
            raise ValueError("coverage row must contain exactly the required columns")

        sample_id = row["sample_id"]
        _require_sample_id(sample_id)
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)

        status = row["status"]
        if not isinstance(status, str) or status not in {"success", "failed"}:
            raise ValueError("status must be success or failed")
        if status == "success":
            if not _is_nonempty_path(row["feature_path"]):
                raise ValueError("success row requires feature_path")
            if not _is_nonempty_path(row["evidence_path"]):
                raise ValueError("success row requires evidence_path")
            success_count += 1
        elif not _is_nonempty_text(row["failure_reason"]):
            raise ValueError("failed row requires failure_reason")

        normalized_rows.append({column: row[column] for column in _COVERAGE_COLUMNS})

    return normalized_rows, {
        "coverage_count": len(normalized_rows),
        "success_count": success_count,
        "failed_count": len(normalized_rows) - success_count,
    }


def _is_nonempty_path(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, os.PathLike):
        return bool(os.fspath(value))
    return False


def _is_nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _write_coverage_outputs(
    output: Path,
    summary: Path,
    rows: list[dict[str, object]],
    counts: dict[str, int],
) -> None:
    csv_temp: Path | None = None
    summary_temp: Path | None = None
    csv_fingerprint: _FileFingerprint | None = None
    summary_fingerprint: _FileFingerprint | None = None
    try:
        csv_temp = _new_temp_path(output)
        with csv_temp.open("w", encoding="utf-8", newline="") as coverage_file:
            writer = csv.DictWriter(coverage_file, fieldnames=_COVERAGE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            coverage_file.flush()
            os.fsync(coverage_file.fileno())

        summary_temp = _new_temp_path(summary)
        with summary_temp.open("w", encoding="utf-8") as summary_file:
            json.dump(counts, summary_file, indent=2, sort_keys=True)
            summary_file.write("\n")
            summary_file.flush()
            os.fsync(summary_file.fileno())

        if _path_exists(output) or _path_exists(summary):
            raise ValueError("coverage output or summary already exists")
        csv_fingerprint = _file_fingerprint(csv_temp)
        os.replace(csv_temp, output)
        csv_temp = None
        summary_fingerprint = _file_fingerprint(summary_temp)
        os.replace(summary_temp, summary)
        summary_temp = None
    except Exception:
        _unlink_if_unchanged(summary, summary_fingerprint)
        _unlink_if_unchanged(output, csv_fingerprint)
        raise
    finally:
        for temporary_path in (csv_temp, summary_temp):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _new_temp_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _file_fingerprint(path: Path) -> _FileFingerprint | None:
    try:
        before = path.stat()
        with path.open("rb") as content:
            digest = hashlib.file_digest(content, "sha256").digest()
        after = path.stat()
    except OSError:
        return None
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        return None
    return (*identity, digest)


def _unlink_if_unchanged(path: Path, fingerprint: _FileFingerprint | None) -> None:
    if fingerprint is not None and _file_fingerprint(path) == fingerprint:
        path.unlink(missing_ok=True)
