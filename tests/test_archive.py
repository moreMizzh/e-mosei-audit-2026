from __future__ import annotations

import os
import io
import pickle
import stat
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e_mosei_audit.archive import ArchiveError, ArchiveMember, SevenZipArchive
from e_mosei_audit.workflow import run_audit


def test_rejects_missing_first_split_volume(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"not a zip")
    (tmp_path / "data.z02").write_bytes(b"second volume")

    archive = SevenZipArchive(archive_path, tmp_path / "7za")

    with pytest.raises(ArchiveError, match=r"data\.z01"):
        archive.validate_volumes()


def _fake_7za(tmp_path, *, integrity_exit_code: int = 0, extraction_exit_code: int = 0):
    executable = tmp_path / "fake_7za.py"
    executable.write_text(
        f"""#!/usr/bin/env python3
import sys

command = sys.argv[1]
if command == "t":
    print("Everything is Ok")
    sys.exit({integrity_exit_code})
elif command == "l":
    print("Path = archive.zip")
    print("----------")
    print("Path = root/file.txt")
    print("Size = 7")
    print("Packed Size = 4")
    print("Attributes = ....A")
elif command == "x":
    sys.stdout.buffer.write(b"payload")
    sys.exit({extraction_exit_code})
else:
    raise SystemExit(2)
"""
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def test_verifies_archive_and_lists_technical_members(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, _fake_7za(tmp_path))

    archive.verify()

    assert archive.list_members() == [
        ArchiveMember(path="root/file.txt", size=7, packed_size=4, is_directory=False)
    ]


def test_rejects_a_failed_archive_integrity_check(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, _fake_7za(tmp_path, integrity_exit_code=7))

    with pytest.raises(ArchiveError, match="integrity check failed"):
        archive.verify()


def test_rejects_an_unavailable_seven_zip_executable(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, tmp_path / "missing-7za")

    with pytest.raises(ArchiveError, match="executable is unavailable"):
        archive.verify()


def test_raises_when_member_extraction_fails(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, _fake_7za(tmp_path, extraction_exit_code=7))

    with pytest.raises(ArchiveError, match="could not read member"):
        archive.read_bytes("root/file.txt")


def test_raises_when_streamed_member_extraction_fails(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, _fake_7za(tmp_path, extraction_exit_code=7))

    with pytest.raises(ArchiveError, match="could not read member"):
        with archive.open_member("root/file.txt") as stream:
            assert stream.read() == b"payload"


def test_reads_an_ordinary_zip_through_real_seven_zip(tmp_path) -> None:
    executable = _real_seven_zip()
    if executable is None:
        pytest.skip("set E_MOSEI_7ZA or prepare .tools/bin/7za to exercise real 7-Zip")

    archive_path = tmp_path / "ordinary.zip"
    with zipfile.ZipFile(archive_path, "w") as archive_file:
        archive_file.writestr("root/file.txt", b"streamed payload")

    archive = SevenZipArchive(archive_path, executable)

    archive.verify()
    assert archive.list_members() == [
        ArchiveMember(path="root/file.txt", size=16, packed_size=16, is_directory=False)
    ]
    assert archive.read_bytes("root/file.txt") == b"streamed payload"
    with archive.open_member("root/file.txt") as stream:
        assert stream.read() == b"streamed payload"


def test_audits_all_artifacts_from_an_ordinary_zip_through_real_seven_zip(tmp_path) -> None:
    executable = _real_seven_zip()
    if executable is None:
        pytest.skip("set E_MOSEI_7ZA or prepare .tools/bin/7za to exercise real 7-Zip")

    prefix = "E题数据"
    video_path = f"{prefix}/附件1-数据集原始多模态样本/videos/video-a/01.mp4"
    label_path = f"{prefix}/附件1-数据集原始多模态样本/label-100.xlsx"
    aligned_path = f"{prefix}/附件2-数据集特征文件/aligned_50.pkl"
    unaligned_path = f"{prefix}/附件2-数据集特征文件/unaligned_50.pkl"
    attachment_three_path = f"{prefix}/附件3-模态缺失特征样本/对齐版本/附件3_01.pkl"
    attachment_four_path = f"{prefix}/附件4-可解释专项视频样本与特征文件/对齐版本/01.pkl"
    archive_path = tmp_path / "ordinary.zip"
    with zipfile.ZipFile(archive_path, "w") as archive_file:
        archive_file.writestr(video_path, _mp4())
        archive_file.writestr(label_path, _xlsx())
        archive_file.writestr(aligned_path, pickle.dumps(_feature_payload()))
        archive_file.writestr(unaligned_path, pickle.dumps(_feature_payload()))
        archive_file.writestr(attachment_three_path, pickle.dumps({"test": {"audio": np.ones((1, 2, 2))}}))
        archive_file.writestr(attachment_four_path, pickle.dumps({"id": "01", "audio": np.ones((2, 2))}))

    output = tmp_path / "audit"
    summary = run_audit(SevenZipArchive(archive_path, executable), output, archive_name="ordinary.zip")

    assert summary["raw_error_count"] == 0
    assert summary["feature_error_count"] == 0
    assert summary["special_record_count"] == 2
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "raw_samples.csv",
        "feature_contract.json",
        "special_samples.csv",
        "audit_report.md",
    }


def _real_seven_zip() -> Path | None:
    configured = os.environ.get("E_MOSEI_7ZA")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).parents[1] / ".tools/bin/7za")
    return next((candidate for candidate in candidates if candidate.is_file() and os.access(candidate, os.X_OK)), None)


def _mp4() -> bytes:
    mvhd = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, 1_000, 2_648)
    return _box(b"moov", _box(b"mvhd", mvhd))


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


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
