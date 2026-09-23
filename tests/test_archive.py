from __future__ import annotations

import os
import stat

import pytest

from e_mosei_audit.archive import ArchiveError, ArchiveMember, SevenZipArchive


def test_rejects_missing_first_split_volume(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"not a zip")
    (tmp_path / "data.z02").write_bytes(b"second volume")

    archive = SevenZipArchive(archive_path, tmp_path / "7za")

    with pytest.raises(ArchiveError, match=r"data\.z01"):
        archive.validate_volumes()


def _fake_7za(tmp_path, *, extraction_exit_code: int = 0):
    executable = tmp_path / "fake_7za.py"
    executable.write_text(
        f"""#!/usr/bin/env python3
import sys

command = sys.argv[1]
if command == "t":
    print("Everything is Ok")
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


def test_raises_when_member_extraction_fails(tmp_path) -> None:
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(b"ordinary archive placeholder")
    archive = SevenZipArchive(archive_path, _fake_7za(tmp_path, extraction_exit_code=7))

    with pytest.raises(ArchiveError, match="could not read member"):
        archive.read_bytes("root/file.txt")
