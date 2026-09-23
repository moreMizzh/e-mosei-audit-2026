"""Attachment 1 label-to-video mapping and lightweight media evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable, Iterable, Mapping

from .archive import ArchiveMember
from .media import parse_mvhd_duration


@dataclass
class RawAuditResult:
    records: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def audit_raw_mapping(
    members: Iterable[ArchiveMember],
    label_rows: Iterable[Mapping[str, object]],
    read_member: Callable[[str], bytes],
) -> RawAuditResult:
    """Match Attachment 1 labels to video members without extracting the archive."""

    result = RawAuditResult()
    labels_by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in label_rows:
        key = _sample_key(row["video_id"], row["clip_id"])
        labels_by_key[key].append(row)

    videos_by_key: dict[str, list[ArchiveMember]] = defaultdict(list)
    for member in members:
        key = _attachment_one_video_key(member)
        if key is not None:
            videos_by_key[key].append(member)

    for key in sorted(labels_by_key):
        rows = labels_by_key[key]
        if len(rows) > 1:
            result.errors.append(f"duplicate label key: {key}")
            result.records.append(_label_record(rows[0], mapping_status="duplicate_label"))
            if key not in videos_by_key:
                result.errors.append(f"label has no video: {key}")
            continue

        row = rows[0]
        videos = videos_by_key.get(key, [])
        if not videos:
            result.errors.append(f"label has no video: {key}")
            result.records.append(_label_record(row, mapping_status="missing_video"))
            continue
        if len(videos) > 1:
            result.errors.append(f"duplicate video key: {key}")
            result.records.append(_label_record(row, mapping_status="duplicate_video"))
            continue

        record = _label_record(row, mapping_status="matched")
        record["member_path"] = videos[0].path
        try:
            record["duration_seconds"] = parse_mvhd_duration(read_member(videos[0].path))
            record["duration_status"] = "parsed"
        except (OSError, ValueError) as error:
            record["duration_seconds"] = None
            record["duration_status"] = f"unreadable: {error}"
            result.warnings.append(f"duration unavailable for {key}: {error}")
        result.records.append(record)

    for key in sorted(set(videos_by_key) - set(labels_by_key)):
        video = videos_by_key[key][0]
        video_id, clip_id = _split_sample_key(key)
        result.errors.append(f"video has no label: {key}")
        result.records.append(
            {
                "sample_id": key,
                "video_id": video_id,
                "clip_id": clip_id,
                "member_path": video.path,
                "mapping_status": "extra_video",
                "duration_seconds": None,
                "duration_status": "not_checked",
            }
        )
    return result


def _attachment_one_video_key(member: ArchiveMember) -> str | None:
    path = PurePosixPath(member.path)
    if member.is_directory or path.suffix.lower() != ".mp4":
        return None
    if not any(part.startswith("附件1-") for part in path.parts):
        return None
    if len(path.parts) < 2:
        return None
    return _sample_key(path.parent.name, path.stem)


def _label_record(row: Mapping[str, object], *, mapping_status: str) -> dict[str, object]:
    video_id = _as_identifier(row["video_id"])
    clip_id = _as_identifier(row["clip_id"])
    record: dict[str, object] = {
        "sample_id": _sample_key(video_id, clip_id),
        "video_id": video_id,
        "clip_id": clip_id,
        "member_path": None,
        "label": row.get("label"),
        "annotation": row.get("annotation"),
        "mapping_status": mapping_status,
        "duration_seconds": None,
        "duration_status": "not_checked",
    }
    if "text" in row:
        record["text"] = row["text"]
    return record


def _as_identifier(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sample_key(video_id: object, clip_id: object) -> str:
    return f"{_as_identifier(video_id)}_{_as_identifier(clip_id)}"


def _split_sample_key(key: str) -> tuple[str, str]:
    return key.rsplit("_", maxsplit=1)
