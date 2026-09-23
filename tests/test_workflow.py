from __future__ import annotations

import csv
import io
import json
import pickle
import struct
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from e_mosei_audit.archive import ArchiveMember
from e_mosei_audit.workflow import run_audit


class _MemoryArchive:
    def __init__(self, members: list[ArchiveMember], contents: dict[str, bytes]) -> None:
        self.members = members
        self.contents = contents
        self.verified = False

    def verify(self) -> None:
        self.verified = True

    def list_members(self) -> list[ArchiveMember]:
        return self.members

    def read_bytes(self, member_path: str) -> bytes:
        return self.contents[member_path]

    @contextmanager
    def open_member(self, member_path: str):
        yield io.BytesIO(self.contents[member_path])


def _member(path: str, size: int = 100) -> ArchiveMember:
    return ArchiveMember(path=path, size=size, packed_size=size - 10, is_directory=False)


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _mp4() -> bytes:
    mvhd = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, 1_000, 2_648)
    return _box(b"moov", _box(b"mvhd", mvhd))


def _xlsx() -> bytes:
    data = io.BytesIO()
    pd.DataFrame(
        [{"video_id": "video-a", "clip_id": "01", "text": "Text", "label": 1.0, "annotation": "Positive"}]
    ).to_excel(data, index=False)
    return data.getvalue()


def _feature_payload() -> dict:
    return {
        split: {
            "id": np.array([f"{split}-1"]),
            "audio": np.ones((1, 2, 2), dtype=np.float32),
            "classification_labels": np.array([1]),
            "regression_labels": np.array([1.0]),
        }
        for split in ("train", "valid", "test")
    }


def test_writes_all_audit_artifacts_from_read_only_archive_members(tmp_path) -> None:
    prefix = "E题数据"
    video_path = f"{prefix}/附件1-数据集原始多模态样本/videos/video-a/01.mp4"
    label_path = f"{prefix}/附件1-数据集原始多模态样本/label-100.xlsx"
    aligned_path = f"{prefix}/附件2-数据集特征文件/aligned_50.pkl"
    unaligned_path = f"{prefix}/附件2-数据集特征文件/unaligned_50.pkl"
    attachment_three_path = f"{prefix}/附件3-模态缺失特征样本/对齐版本/附件3_01.pkl"
    attachment_four_path = f"{prefix}/附件4-可解释专项视频样本与特征文件/对齐版本/01.pkl"
    members = [
        _member(video_path),
        _member(label_path),
        _member(aligned_path),
        _member(unaligned_path),
        _member(attachment_three_path),
        _member(attachment_four_path),
    ]
    contents = {
        video_path: _mp4(),
        label_path: _xlsx(),
        aligned_path: pickle.dumps(_feature_payload()),
        unaligned_path: pickle.dumps(_feature_payload()),
        attachment_three_path: pickle.dumps({"test": {"audio": np.ones((1, 2, 2))}}),
        attachment_four_path: pickle.dumps({"id": "01", "audio": np.ones((2, 2))}),
    }
    archive = _MemoryArchive(members, contents)
    output = tmp_path / "audit"

    summary = run_audit(archive, output, archive_name="fixture.zip")

    assert archive.verified is True
    assert summary["raw_error_count"] == 0
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "raw_samples.csv",
        "feature_contract.json",
        "special_samples.csv",
        "audit_report.md",
    }
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["member_count"] == 6
    with (output / "raw_samples.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["mapping_status"] == "matched"
    assert len(list(csv.DictReader((output / "special_samples.csv").open(encoding="utf-8")))) == 2


def test_refuses_to_overwrite_an_existing_output_directory(tmp_path) -> None:
    output = tmp_path / "audit"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_audit(_MemoryArchive([], {}), output, archive_name="fixture.zip")
