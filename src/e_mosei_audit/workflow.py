"""End-to-end orchestration for the read-only E-problem data audit."""

from __future__ import annotations

import csv
import io
import json
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import pandas as pd

from .archive import ArchiveMember
from .features import audit_feature_dataset
from .raw import audit_raw_mapping
from .special import audit_special_payload


class ArchiveReader(Protocol):
    def verify(self) -> None: ...

    def list_members(self) -> list[ArchiveMember]: ...

    def read_bytes(self, member_path: str) -> bytes: ...

    def open_member(self, member_path: str) -> Iterator[io.BufferedReader]: ...


def run_audit(archive: ArchiveReader, output_dir: Path, *, archive_name: str) -> dict[str, object]:
    """Audit one archive and create the five stable, derived evidence artifacts."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"audit output directory already exists: {output_dir}")

    archive.verify()
    members = archive.list_members()
    file_members = [member for member in members if not member.is_directory]

    label_member = _single_member(file_members, "label-100.xlsx")
    label_rows = _read_label_rows(archive.read_bytes(label_member.path))
    raw_result = audit_raw_mapping(file_members, label_rows, archive.read_bytes)

    feature_contracts: dict[str, object] = {}
    for filename in ("aligned_50.pkl", "unaligned_50.pkl"):
        member = _single_member(file_members, filename)
        with archive.open_member(member.path) as stream:
            payload = pickle.load(stream)
        feature_contracts[filename] = audit_feature_dataset(payload, source_name=filename)
        del payload

    special_records: list[dict[str, object]] = []
    special_errors: list[str] = []
    for member in file_members:
        attachment = _special_attachment(member.path)
        if attachment is None or PurePosixPath(member.path).suffix.lower() != ".pkl":
            continue
        with archive.open_member(member.path) as stream:
            payload = pickle.load(stream)
        result = audit_special_payload(
            payload,
            attachment=attachment,
            version=_special_version(member.path),
            source_file=PurePosixPath(member.path).name,
        )
        special_records.extend(result.records)
        special_errors.extend(result.errors)
        del payload

    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "manifest.json",
        {
            "archive_name": archive_name,
            "member_count": len(members),
            "file_count": len(file_members),
            "members": [
                {
                    "path": member.path,
                    "size": member.size,
                    "packed_size": member.packed_size,
                    "is_directory": member.is_directory,
                }
                for member in members
            ],
        },
    )
    _write_csv(output_dir / "raw_samples.csv", raw_result.records, _RAW_COLUMNS)
    _write_json(output_dir / "feature_contract.json", feature_contracts)
    _write_special_csv(output_dir / "special_samples.csv", special_records)

    summary = {
        "archive_name": archive_name,
        "member_count": len(members),
        "raw_error_count": len(raw_result.errors),
        "raw_warning_count": len(raw_result.warnings),
        "feature_sources": list(feature_contracts),
        "special_record_count": len(special_records),
        "special_error_count": len(special_errors),
    }
    (output_dir / "audit_report.md").write_text(
        _render_report(summary, raw_result.errors, raw_result.warnings, special_errors), encoding="utf-8"
    )
    return summary


_RAW_COLUMNS = (
    "sample_id",
    "video_id",
    "clip_id",
    "member_path",
    "text",
    "label",
    "annotation",
    "mapping_status",
    "duration_seconds",
    "duration_status",
)


def _single_member(members: list[ArchiveMember], filename: str) -> ArchiveMember:
    matched = [member for member in members if PurePosixPath(member.path).name == filename]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one archive member named {filename!r}, found {len(matched)}")
    return matched[0]


def _read_label_rows(content: bytes) -> list[dict[str, object]]:
    frame = pd.read_excel(io.BytesIO(content), dtype={"video_id": str, "clip_id": str})
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"label-100.xlsx is missing required columns: {', '.join(missing)}")
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def _special_attachment(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if any(part.startswith("附件3-") for part in parts):
        return "attachment3"
    if any(part.startswith("附件4-") for part in parts):
        return "attachment4"
    return None


def _special_version(path: str) -> str:
    parts = PurePosixPath(path).parts
    if "未对齐版本" in parts:
        return "unaligned"
    if "对齐版本" in parts:
        return "aligned"
    return "unknown"


def _write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[Mapping[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_special_csv(path: Path, records: list[Mapping[str, object]]) -> None:
    fields = ("attachment", "version", "source_file", "sample_id", "layout", "fields_json")
    flattened = [
        {
            **{field: record.get(field) for field in fields if field != "fields_json"},
            "fields_json": json.dumps(record["fields"], ensure_ascii=False, sort_keys=True, default=_json_default),
        }
        for record in records
    ]
    _write_csv(path, flattened, fields)


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _render_report(
    summary: Mapping[str, object], raw_errors: list[str], raw_warnings: list[str], special_errors: list[str]
) -> str:
    lines = [
        "# E 题数据审计报告",
        "",
        f"- 归档：`{summary['archive_name']}`",
        f"- 归档成员：{summary['member_count']}",
        f"- 附件 1 映射错误：{summary['raw_error_count']}",
        f"- 附件 1 时长警告：{summary['raw_warning_count']}",
        f"- 特征文件：{', '.join(summary['feature_sources'])}",
        f"- 专项样本记录：{summary['special_record_count']}",
        f"- 专项样本错误：{summary['special_error_count']}",
        "",
        "## 解释边界",
        "",
        "专项样本中的连续全零位置仅记录为数值证据。除非同一文件提供独立长度或掩码字段，报告不会将其判定为赛题注入的模态缺失。",
    ]
    for heading, messages in (("附件 1 错误", raw_errors), ("附件 1 警告", raw_warnings), ("专项样本错误", special_errors)):
        if messages:
            lines.extend(["", f"## {heading}", ""])
            lines.extend(f"- {message}" for message in messages)
    return "\n".join(lines) + "\n"
