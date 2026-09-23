from __future__ import annotations

import struct

from e_mosei_audit.archive import ArchiveMember
from e_mosei_audit.raw import audit_raw_mapping


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _video(duration: int = 3_000) -> bytes:
    mvhd = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, 1_000, duration)
    return _box(b"moov", _box(b"mvhd", mvhd))


def _member(path: str) -> ArchiveMember:
    return ArchiveMember(path=path, size=100, packed_size=80, is_directory=False)


def test_matches_labels_to_attachment_one_videos_and_parses_duration() -> None:
    path = "E题数据/附件1-数据集原始多模态样本/videos/video-a/01.mp4"
    result = audit_raw_mapping(
        [_member(path)],
        [{"video_id": "video-a", "clip_id": "01", "label": 1.5, "annotation": "Positive"}],
        lambda member_path: _video() if member_path == path else b"",
    )

    assert result.errors == []
    assert result.records == [
        {
            "sample_id": "video-a_01",
            "video_id": "video-a",
            "clip_id": "01",
            "member_path": path,
            "label": 1.5,
            "annotation": "Positive",
            "mapping_status": "matched",
            "duration_seconds": 3.0,
            "duration_status": "parsed",
        }
    ]


def test_reports_missing_extra_and_duplicate_label_keys() -> None:
    member_path = "E题数据/附件1-数据集原始多模态样本/videos/extra/07.mp4"
    result = audit_raw_mapping(
        [_member(member_path)],
        [
            {"video_id": "missing", "clip_id": "02"},
            {"video_id": "missing", "clip_id": "02"},
        ],
        lambda _: _video(),
    )

    assert {record["mapping_status"] for record in result.records} == {
        "duplicate_label",
        "extra_video",
    }
    assert result.errors == [
        "duplicate label key: missing_02",
        "label has no video: missing_02",
        "video has no label: extra_07",
    ]
