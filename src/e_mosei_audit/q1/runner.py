"""Audit-backed Question 1 feature extraction and evidence generation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from e_mosei_audit.archive import ArchiveMember, SevenZipArchive
from e_mosei_audit.q1.config import Q1Config
from e_mosei_audit.q1.contracts import SampleFeatures, write_coverage
from e_mosei_audit.q1.extractors import (
    AudioExtractor,
    TextEncoder,
    VisionExtractor,
    WordAligner,
    WordInterval,
    build_bert_encoder,
    build_mediapipe_extractor,
    build_opensmile_extractor,
    build_whisperx_aligner,
    text_slot_features,
)
from e_mosei_audit.q1.media import MediaError, decode_member
from e_mosei_audit.q1.timeline import make_slots, pool_intervals, pool_moments
from e_mosei_audit.workflow import ArchiveReader


_AUDIT_COLUMNS = (
    "sample_id",
    "video_id",
    "clip_id",
    "member_path",
    "text",
    "mapping_status",
    "duration_seconds",
    "duration_status",
)
_FEATURE_CONTRACT = {
    "slot_count": 50,
    "features": {
        "text": {"shape": [50, 768], "dtype": "float16"},
        "audio": {"shape": [50, 50], "dtype": "float16"},
        "vision": {"shape": [50, 56], "dtype": "float16"},
    },
    "slots": {"shape": [50, 2], "dtype": "float64"},
    "masks": {"shape": [50], "dtype": "bool"},
    "vision_channels": {
        "face_presence": "1.0 denotes a detected face, not a detector confidence; frames without a detected face produce no visual row."
    },
}
_FFMPEG_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Q1Extractors:
    """Injected feature adapters, kept small so extraction remains testable."""

    aligner: WordAligner
    text: TextEncoder
    audio: AudioExtractor
    vision: VisionExtractor


def run_q1(
    config: Q1Config,
    *,
    archive: ArchiveReader | None = None,
    extractors: Q1Extractors | None = None,
    limit: int | None = None,
) -> dict[str, int | str]:
    """Extract aligned features from selected audited Attachment 1 samples."""

    _validate_output_target(config.output_dir)
    selected_rows = _select_audit_rows(config.audit_dir, limit)
    ffmpeg_preflight = _preflight_ffmpeg(config.ffmpeg)

    active_archive = archive or SevenZipArchive(config.archive, config.seven_zip)
    archive_tool = _preflight_archive_tool(config.seven_zip, injected=archive is not None)
    active_archive.verify()
    members = active_archive.list_members()
    member_paths = _member_paths(members)
    active_extractors = extractors or _build_default_extractors(config)
    reproducibility = _reproducibility_metadata(
        config,
        limit=limit,
        source_archive=_source_archive_fingerprint(config.archive),
        ffmpeg_preflight=ffmpeg_preflight,
        archive_tool=archive_tool,
        injected_extractors=extractors is not None,
    )

    config.output_dir.mkdir()
    coverage_rows: list[dict[str, object]] = []
    log_lines = ["Q1 extraction run"]
    first_success: dict[str, object] | None = None

    with TemporaryDirectory(dir=config.output_dir) as temporary_directory:
        work_root = Path(temporary_directory)
        for row in selected_rows:
            coverage, successful_evidence = _extract_audited_row(
                row,
                archive=active_archive,
                member_paths=member_paths,
                extractors=active_extractors,
                config=config,
                work_root=work_root,
            )
            coverage_rows.append(coverage)
            if coverage["status"] == "success":
                log_lines.append(f"{coverage['sample_id']}: success")
                if first_success is None:
                    first_success = successful_evidence
            else:
                log_lines.append(
                    f"{coverage['sample_id']}: failed: {coverage['failure_reason']}"
                )

    counts = write_coverage(
        config.output_dir / "q1_samples.csv",
        coverage_rows,
        expected_count=len(selected_rows),
    )
    _write_json(
        config.output_dir / "feature_contract.json",
        {
            **_FEATURE_CONTRACT,
            "extractors": _extractor_identity(active_extractors),
            "reproducibility": reproducibility,
            "output_counts": counts,
        },
    )
    _write_json(
        config.output_dir / "run_manifest.json",
        {
            "audit_dir": str(config.audit_dir),
            "audit_csv": str(config.audit_dir / "raw_samples.csv"),
            "archive": str(config.archive),
            "seven_zip": str(config.seven_zip),
            "ffmpeg": str(config.ffmpeg),
            "model_cache": str(config.model_cache),
            "output_dir": str(config.output_dir),
            "archive_member_count": len(members),
            "selected_row_count": len(selected_rows),
            **counts,
            "output_counts": counts,
            "reproducibility": reproducibility,
        },
    )
    (config.output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    (config.output_dir / "typical_sample.md").write_text(
        _render_typical_sample(first_success), encoding="utf-8"
    )
    return counts


def _validate_output_target(output_dir: object) -> Path:
    if not isinstance(output_dir, Path):
        raise ValueError("output_dir must be a Path")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Q1 output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ValueError(f"output_dir parent must be an existing directory: {output_dir.parent}")
    return output_dir


def _preflight_ffmpeg(ffmpeg: object) -> dict[str, object]:
    if (
        not isinstance(ffmpeg, Path)
        or not ffmpeg.is_file()
        or not os.access(ffmpeg, os.X_OK)
    ):
        raise RuntimeError(
            f"ffmpeg preflight requires an executable regular file: {ffmpeg}"
        )
    command = [str(ffmpeg), "-version"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"ffmpeg preflight timed out after {_FFMPEG_TIMEOUT_SECONDS:g} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(f"ffmpeg preflight could not run: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        raise RuntimeError(f"ffmpeg preflight failed: {detail}")
    version_output = (completed.stdout or completed.stderr).strip()
    version = version_output.splitlines()[0] if version_output else "available (no version text)"
    return {"path": str(ffmpeg), "command": command, "version": version}


def _preflight_archive_tool(seven_zip: Path, *, injected: bool) -> dict[str, object]:
    if injected:
        return {
            "path": str(seven_zip),
            "command": None,
            "version": "not-applicable: injected archive",
        }
    if not seven_zip.is_file() or not os.access(seven_zip, os.X_OK):
        raise RuntimeError(
            f"7-Zip preflight requires an executable regular file: {seven_zip}"
        )
    command = [str(seven_zip), "i"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("7-Zip preflight timed out") from error
    except OSError as error:
        raise RuntimeError(f"7-Zip preflight could not run: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        raise RuntimeError(f"7-Zip preflight failed: {detail}")
    version_output = (completed.stdout or completed.stderr).strip()
    version = version_output.splitlines()[0] if version_output else "available (no version text)"
    return {"path": str(seven_zip), "command": command, "version": version}


def _source_archive_fingerprint(archive: Path) -> dict[str, object]:
    if not archive.is_file():
        raise RuntimeError(f"source archive fingerprint requires a regular file: {archive}")
    split_volumes = sorted(
        archive.parent.glob(f"{archive.stem}.z[0-9][0-9]"),
        key=lambda candidate: int(candidate.suffix[2:]),
    )
    return {
        "zip": _file_fingerprint(archive),
        "split_volumes": [_file_fingerprint(volume) for volume in split_volumes],
    }


def _file_fingerprint(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        size = path.stat().st_size
    except OSError as error:
        raise RuntimeError(f"could not fingerprint source file {path}: {error}") from error
    return {"path": str(path), "size": size, "sha256": digest}


def _reproducibility_metadata(
    config: Q1Config,
    *,
    limit: int | None,
    source_archive: Mapping[str, object],
    ffmpeg_preflight: Mapping[str, object],
    archive_tool: Mapping[str, object],
    injected_extractors: bool,
) -> dict[str, object]:
    return {
        "source_archive": dict(source_archive),
        "logical_invocation": {
            "operation": "run_q1",
            "mode": "full" if limit is None else "smoke",
            "limit": limit,
            "output_dir": str(config.output_dir),
        },
        "tool_commands": {
            "archive_verify": [str(config.seven_zip), "t", str(config.archive)],
            "archive_list": [str(config.seven_zip), "l", "-slt", str(config.archive)],
            "ffmpeg_preflight": ffmpeg_preflight["command"],
            "audio_decode": [
                str(config.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "<archive-member.mp4>",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "<audio.wav>",
            ],
            "frame_decode": [
                str(config.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "<archive-member.mp4>",
                "-vf",
                "fps=10",
                "<frames/frame_%06d.png>",
            ],
        },
        "tool_versions": {
            "ffmpeg": dict(ffmpeg_preflight),
            "seven_zip": dict(archive_tool),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            package: _package_version(package)
            for package in (
                "numpy",
                "transformers",
                "torch",
                "whisperx",
                "nltk",
                "opensmile",
                "mediapipe",
            )
        },
        "models": {
            "whisperx_alignment": {
                "id": "facebook/wav2vec2-base-960h",
                "local_cache": str(config.model_cache),
            },
            "nltk_punkt_tab": {
                "id": "tokenizers/punkt_tab/english.pickle",
                "local_cache": str(config.model_cache / "nltk_data"),
            },
            "bert": {"id": "bert-base-uncased", "local_cache": str(config.model_cache)},
            "mediapipe_face_landmarker": {
                "id": "face_landmarker.task",
                "local_cache": str(config.model_cache),
            },
            "extractors": "injected" if injected_extractors else "configured local models",
        },
        "parameters": {
            "slot_count": 50,
            "slots": "50 contiguous physical intervals spanning audited duration",
            "text": {"dimensions": 768, "model": "bert-base-uncased"},
            "audio": {
                "extractor": "OpenSMILE eGeMAPSv02 LowLevelDescriptors",
                "window_dimensions": 25,
                "pooled_dimensions": 50,
                "pooling": "overlap-weighted mean and population standard deviation",
                "decode_sample_rate_hz": 16000,
                "decode_channels": 1,
            },
            "vision": {
                "extractor": "MediaPipe Face Landmarker",
                "fps": 10,
                "dimensions": 56,
                "channel_layout": [
                    *[f"blendshape_{index}" for index in range(52)],
                    "face_area",
                    "face_center_x",
                    "face_center_y",
                    "face_presence",
                ],
            },
        },
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _select_audit_rows(audit_dir: Path, limit: int | None) -> list[dict[str, str]]:
    if isinstance(limit, bool) or (limit is not None and not isinstance(limit, int)):
        raise ValueError("limit must be a positive integer")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    audit_path = audit_dir / "raw_samples.csv"
    try:
        with audit_path.open("r", encoding="utf-8", newline="") as audit_file:
            reader = csv.DictReader(audit_file)
            if reader.fieldnames is None:
                raise ValueError("raw_samples.csv must have a header")
            missing = [field for field in _AUDIT_COLUMNS if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"raw_samples.csv is missing required column: {missing[0]}")
            rows = [{field: row.get(field, "") for field in _AUDIT_COLUMNS} for row in reader]
    except OSError as error:
        raise ValueError(f"could not read audit raw_samples.csv: {error}") from error

    sample_ids: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id.strip():
            raise ValueError("raw_samples.csv has an empty sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"raw_samples.csv has duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)

    if limit is None:
        if len(rows) != 100:
            raise ValueError(f"full Q1 extraction requires exactly 100 audit rows, found {len(rows)}")
        return rows
    if limit > len(rows):
        raise ValueError(f"limit {limit} exceeds available audit rows {len(rows)}")
    return rows[:limit]


def _member_paths(members: Sequence[ArchiveMember]) -> set[str]:
    return {member.path for member in members if not member.is_directory}


def _build_default_extractors(config: Q1Config) -> Q1Extractors:
    return Q1Extractors(
        aligner=build_whisperx_aligner(config.model_cache),
        text=build_bert_encoder(config.model_cache),
        audio=build_opensmile_extractor(),
        vision=build_mediapipe_extractor(config.model_cache),
    )


def _extract_audited_row(
    row: Mapping[str, str],
    *,
    archive: ArchiveReader,
    member_paths: set[str],
    extractors: Q1Extractors,
    config: Q1Config,
    work_root: Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    sample_id = row["sample_id"]
    eligible, duration_or_reason = _eligibility(row, member_paths)
    if not eligible:
        return _failed_coverage(row, str(duration_or_reason)), None
    duration = float(duration_or_reason)

    try:
        transcript = _nonempty_text(row["text"], "text")
        feature_name = _artifact_name(sample_id, ".npz")
        evidence_name = _artifact_name(sample_id, ".json")
        feature_relative = Path("features") / feature_name
        evidence_relative = Path("alignment_evidence") / evidence_name
        feature_path = config.output_dir / feature_relative
        evidence_path = config.output_dir / evidence_relative
        slots = make_slots(duration)

        mp4_bytes = archive.read_bytes(row["member_path"])
        with decode_member(config.ffmpeg, mp4_bytes, work_root) as media:
            words = extractors.aligner.align(media.wav_path, transcript, duration)
            text_embeddings, text_intervals = extractors.text.encode(transcript, words)
            text, text_mask = text_slot_features(text_embeddings, text_intervals, slots)

            audio_values, audio_starts, audio_ends = extractors.audio.extract(media.wav_path)
            audio, audio_mask = pool_moments(audio_values, audio_starts, audio_ends, slots)

            vision_values, vision_starts, vision_ends = extractors.vision.extract(
                media.frame_dir
            )
            vision, vision_mask = pool_intervals(
                vision_values, vision_starts, vision_ends, slots
            )
            decoded_frame_count = len(tuple(media.frame_dir.glob("*.png")))

        evidence = _evidence(
            sample_id=sample_id,
            transcript=transcript,
            slots=slots,
            words=words,
            token_intervals=text_intervals,
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask,
            audio_starts=audio_starts,
            audio_ends=audio_ends,
            vision_starts=vision_starts,
            vision_ends=vision_ends,
            audio_window_count=len(audio_values),
            visual_frame_count=len(vision_values),
            decoded_frame_count=decoded_frame_count,
        )
        _write_json(evidence_path, evidence)
        SampleFeatures(
            sample_id=sample_id,
            text=text,
            audio=audio,
            vision=vision,
            slots=slots,
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask,
        ).write(feature_path)
        return (
            _success_coverage(
                row,
                duration=duration,
                feature_path=feature_relative,
                evidence_path=evidence_relative,
            ),
            evidence,
        )
    except (MediaError, RuntimeError, ValueError, OSError) as error:
        for path in (feature_path if "feature_path" in locals() else None, evidence_path if "evidence_path" in locals() else None):
            if path is not None:
                path.unlink(missing_ok=True)
        return _failed_coverage(row, str(error)), None


def _eligibility(row: Mapping[str, str], member_paths: set[str]) -> tuple[bool, float | str]:
    if row["mapping_status"] != "matched":
        return False, "mapping_status must be matched"
    if row["duration_status"] != "parsed":
        return False, "duration_status must be parsed"
    try:
        duration = float(row["duration_seconds"])
    except (TypeError, ValueError, OverflowError):
        return False, "duration_seconds must be finite and positive"
    if not math.isfinite(duration) or duration <= 0:
        return False, "duration_seconds must be finite and positive"
    member_path = row["member_path"]
    if not member_path or member_path not in member_paths:
        return False, "member_path is not an exact listed archive member"
    for field in ("video_id", "clip_id"):
        if not row[field].strip():
            return False, f"{field} must be non-empty"
    return True, duration


def _success_coverage(
    row: Mapping[str, str], *, duration: float, feature_path: Path, evidence_path: Path
) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "clip_id": row["clip_id"],
        "member_path": row["member_path"],
        "duration_seconds": duration,
        "status": "success",
        "feature_path": feature_path.as_posix(),
        "evidence_path": evidence_path.as_posix(),
        "text_shape": "(50, 768)",
        "audio_shape": "(50, 50)",
        "vision_shape": "(50, 56)",
        "failure_reason": "",
    }


def _failed_coverage(row: Mapping[str, str], reason: str) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "clip_id": row["clip_id"],
        "member_path": row["member_path"],
        "duration_seconds": row["duration_seconds"],
        "status": "failed",
        "feature_path": "",
        "evidence_path": "",
        "text_shape": "",
        "audio_shape": "",
        "vision_shape": "",
        "failure_reason": reason or "unspecified extraction failure",
    }


def _artifact_name(sample_id: str, suffix: str) -> str:
    if (
        not sample_id
        or sample_id in {".", ".."}
        or Path(sample_id).name != sample_id
        or "/" in sample_id
        or "\\" in sample_id
    ):
        raise ValueError("sample_id is unsafe for output artifact paths")
    return f"{sample_id}{suffix}"


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _evidence(
    *,
    sample_id: str,
    transcript: str,
    slots: np.ndarray,
    words: Sequence[WordInterval],
    token_intervals: Sequence[WordInterval],
    text_mask: np.ndarray,
    audio_mask: np.ndarray,
    vision_mask: np.ndarray,
    audio_starts: object,
    audio_ends: object,
    vision_starts: object,
    vision_ends: object,
    audio_window_count: int,
    visual_frame_count: int,
    decoded_frame_count: int,
) -> dict[str, object]:
    original_words = _text_interval_records(transcript, words, slots, key="word")
    tokens = _text_interval_records(transcript, token_intervals, slots, key="text")
    audio_windows = _time_interval_records(audio_starts, audio_ends, slots)
    visual_rows = _time_interval_records(
        vision_starts, vision_ends, slots, include_frame_index=True
    )
    face_detection_rate = (
        visual_frame_count / decoded_frame_count if decoded_frame_count else 0.0
    )
    return {
        "sample_id": sample_id,
        "transcript": transcript,
        "slots": slots.tolist(),
        "original_words": original_words,
        "token_intervals": tokens,
        "masks": {
            "text": text_mask.tolist(),
            "audio": audio_mask.tolist(),
            "vision": vision_mask.tolist(),
        },
        "audio_window_count": audio_window_count,
        "visual_frame_count": visual_frame_count,
        "audio_windows": audio_windows,
        "detected_visual_rows": visual_rows,
        "decoded_frame_count": decoded_frame_count,
        "detected_face_frame_count": visual_frame_count,
        "face_detection_rate": face_detection_rate,
        "face_presence": {
            "name": "face_presence",
            "meaning": "1.0 denotes a detected face, not a detector confidence; frames without a detected face produce no visual row.",
        },
    }


def _text_interval_records(
    transcript: str, intervals: Sequence[WordInterval], slots: np.ndarray, *, key: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for interval in intervals:
        _validate_transcript_span(transcript, interval)
        records.append(
            {
                key: transcript[interval.char_start : interval.char_end],
                "char_start": interval.char_start,
                "char_end": interval.char_end,
                "start": interval.start,
                "end": interval.end,
                "overlap_slot_indexes": _overlap_slot_indexes(
                    interval.start, interval.end, slots
                ),
            }
        )
    return records


def _time_interval_records(
    starts: object,
    ends: object,
    slots: np.ndarray,
    *,
    include_frame_index: bool = False,
) -> list[dict[str, object]]:
    start_values = np.asarray(starts, dtype=np.float64)
    end_values = np.asarray(ends, dtype=np.float64)
    if start_values.ndim != 1 or end_values.ndim != 1 or start_values.shape != end_values.shape:
        raise ValueError("traceability timestamps must be matching one-dimensional arrays")
    records: list[dict[str, object]] = []
    for start, end in zip(start_values, end_values, strict=True):
        if not math.isfinite(float(start)) or not math.isfinite(float(end)) or end <= start:
            raise ValueError("traceability timestamps must be finite positive intervals")
        record: dict[str, object] = {
            "start": float(start),
            "end": float(end),
            "overlap_slot_indexes": _overlap_slot_indexes(float(start), float(end), slots),
        }
        if include_frame_index:
            record["frame_index"] = int(round(float(start) * 10.0))
        records.append(record)
    return records


def _validate_transcript_span(transcript: str, interval: WordInterval) -> None:
    if interval.char_end > len(transcript):
        raise ValueError("word interval character span exceeds the original transcript")


def _overlap_slot_indexes(start: float, end: float, slots: np.ndarray) -> list[int]:
    return [
        index
        for index, (slot_start, slot_end) in enumerate(slots)
        if min(end, float(slot_end)) > max(start, float(slot_start))
    ]


def _extractor_identity(extractors: Q1Extractors) -> dict[str, str]:
    return {
        name: f"{type(adapter).__module__}.{type(adapter).__qualname__}"
        for name, adapter in (
            ("aligner", extractors.aligner),
            ("text", extractors.text),
            ("audio", extractors.audio),
            ("vision", extractors.vision),
        )
    }


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_typical_sample(evidence: Mapping[str, object] | None) -> str:
    if evidence is None:
        return "# Typical Q1 sample\n\nNo successful sample was extracted.\n"
    slots = _evidence_list(evidence, "slots")
    tokens = _evidence_list(evidence, "token_intervals")
    visual_rows = _evidence_list(evidence, "detected_visual_rows")
    lines = [
        "# Typical Q1 sample",
        "",
        f"- Sample: `{evidence['sample_id']}`",
        f"- Transcript: {evidence['transcript']}",
        f"- Audio windows: {evidence['audio_window_count']}",
        f"- Decoded frames: {evidence['decoded_frame_count']}",
        f"- Detected face frames: {evidence['detected_face_frame_count']}",
        f"- Face detection rate: {float(evidence['face_detection_rate']):.6f}",
        "",
        "| Slot | Start (s) | End (s) | Text fragments | Detected frame indexes |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for index, bounds in enumerate(slots):
        start, end = _slot_bounds(bounds)
        fragments = _slot_text_fragments(tokens, index)
        frame_indexes = _slot_frame_indexes(visual_rows, index)
        lines.append(
            "| "
            f"{index} | {start:.6f} | {end:.6f} | "
            f"{_markdown_cell(', '.join(fragments))} | "
            f"{', '.join(frame_indexes)} |"
        )
    return "\n".join(lines) + "\n"


def _evidence_list(evidence: Mapping[str, object], field: str) -> list[object]:
    value = evidence.get(field)
    if not isinstance(value, list):
        raise ValueError(f"typical sample evidence is missing list field: {field}")
    return value


def _slot_bounds(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("typical sample slot must contain start and end")
    start = float(value[0])
    end = float(value[1])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("typical sample slot must be finite and positive")
    return start, end


def _slot_text_fragments(tokens: list[object], slot_index: int) -> list[str]:
    fragments: list[str] = []
    for token in tokens:
        if not isinstance(token, Mapping):
            raise ValueError("typical sample token interval must be a mapping")
        indexes = token.get("overlap_slot_indexes")
        text = token.get("text")
        if not isinstance(indexes, list) or not isinstance(text, str):
            raise ValueError("typical sample token interval is invalid")
        if slot_index in indexes:
            fragments.append(text)
    return fragments


def _slot_frame_indexes(visual_rows: list[object], slot_index: int) -> list[str]:
    indexes: list[str] = []
    for visual_row in visual_rows:
        if not isinstance(visual_row, Mapping):
            raise ValueError("typical sample visual row must be a mapping")
        slots = visual_row.get("overlap_slot_indexes")
        frame_index = visual_row.get("frame_index")
        if not isinstance(slots, list) or not isinstance(frame_index, int):
            raise ValueError("typical sample visual row is invalid")
        if slot_index in slots:
            indexes.append(str(frame_index))
    return indexes


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
