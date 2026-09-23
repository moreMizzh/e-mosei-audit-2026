"""Read-only access to ordinary and split ZIP archives through 7-Zip."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


class ArchiveError(RuntimeError):
    """Raised when an archive cannot safely be inspected."""


@dataclass(frozen=True)
class ArchiveMember:
    """A member listed by 7-Zip without extracting it to disk."""

    path: str
    size: int
    packed_size: int
    is_directory: bool


@dataclass(frozen=True)
class SevenZipArchive:
    """A split-ZIP input and its explicit 7-Zip executable dependency."""

    archive_path: Path
    executable: Path

    def validate_volumes(self) -> None:
        if not self.archive_path.is_file():
            raise ArchiveError(f"archive does not exist: {self.archive_path}")

        split_parts = self._split_parts()
        if not split_parts:
            return

        highest_index = max(split_parts)
        for index in range(1, highest_index + 1):
            expected = self.archive_path.with_suffix(f".z{index:02d}")
            if index not in split_parts or not expected.is_file():
                raise ArchiveError(f"missing split volume: {expected}")

    def verify(self) -> None:
        """Ask 7-Zip to validate every archive volume before reading members."""

        self.validate_volumes()
        completed = self._run("t", str(self.archive_path))
        if completed.returncode != 0 or "Everything is Ok" not in completed.stdout:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown 7-Zip failure"
            raise ArchiveError(f"archive integrity check failed: {message}")

    def list_members(self) -> list[ArchiveMember]:
        """List archive members from 7-Zip's stable technical output."""

        self.validate_volumes()
        completed = self._run("l", "-slt", str(self.archive_path))
        if completed.returncode != 0:
            raise ArchiveError(self._command_error("could not list archive members", completed))
        return _parse_technical_listing(completed.stdout)

    def read_bytes(self, member_path: str) -> bytes:
        """Read a small member without creating an extracted file."""

        self.validate_volumes()
        completed = subprocess.run(
            [str(self._executable()), "x", "-so", str(self.archive_path), member_path],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip() or "unknown 7-Zip failure"
            raise ArchiveError(f"could not read member {member_path!r}: {message}")
        return completed.stdout

    def open_member(self, member_path: str) -> Iterator[BinaryIO]:
        """Yield a member's stdout stream; callers must consume it completely."""

        self.validate_volumes()
        process = subprocess.Popen(
            [str(self._executable()), "x", "-so", str(self.archive_path), member_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise ArchiveError("could not create 7-Zip member stream")

        return _MemberStream(process, member_path)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self._executable()), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def _executable(self) -> Path:
        if not self.executable.is_file() or not self.executable.stat().st_mode & 0o111:
            raise ArchiveError(f"7-Zip executable is unavailable: {self.executable}")
        return self.executable

    @staticmethod
    def _command_error(prefix: str, completed: subprocess.CompletedProcess[str]) -> str:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown 7-Zip failure"
        return f"{prefix}: {message}"

    def _split_parts(self) -> set[int]:
        stem = self.archive_path.stem
        indices: set[int] = set()
        for candidate in self.archive_path.parent.glob(f"{stem}.z[0-9][0-9]"):
            try:
                indices.add(int(candidate.suffix[2:]))
            except ValueError:
                continue
        return indices


class _MemberStream:
    """Context manager that reports a 7-Zip extraction failure on exit."""

    def __init__(self, process: subprocess.Popen[bytes], member_path: str) -> None:
        self._process = process
        self._member_path = member_path

    def __enter__(self) -> BinaryIO:
        assert self._process.stdout is not None
        return self._process.stdout

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._process.stdout.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = self._process.wait()
        if exc_type is None and returncode != 0:
            raise ArchiveError(f"could not read member {self._member_path!r}: {stderr or 'unknown 7-Zip failure'}")
        return False


def _parse_technical_listing(output: str) -> list[ArchiveMember]:
    members: list[ArchiveMember] = []
    current: dict[str, str] = {}
    in_members = False

    def flush() -> None:
        if "Path" not in current:
            return
        attributes = current.get("Attributes", "")
        members.append(
            ArchiveMember(
                path=current["Path"],
                size=int(current.get("Size", "0")),
                packed_size=int(current.get("Packed Size", "0")),
                is_directory=attributes.startswith("D"),
            )
        )

    for line in output.splitlines():
        if line == "----------":
            in_members = True
            continue
        if not in_members:
            continue
        if not line.strip():
            flush()
            current = {}
            continue
        key, separator, value = line.partition(" = ")
        if not separator:
            continue
        if key == "Path" and current:
            flush()
            current = {}
        current[key] = value
    flush()
    return members
