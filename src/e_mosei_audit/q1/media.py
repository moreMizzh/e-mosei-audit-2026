"""Ephemeral decoding of one raw video member for Question 1."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


class MediaError(RuntimeError):
    """Raised when an input video cannot be decoded into required media."""


@dataclass(frozen=True)
class DecodedMedia:
    wav_path: Path
    frame_dir: Path


@contextmanager
def decode_member(
    ffmpeg: Path,
    mp4_bytes: bytes,
    work_root: Path,
    *,
    timeout_seconds: float = 60.0,
) -> Iterator[DecodedMedia]:
    """Decode audio and ten-fps frames into a temporary workspace."""

    _validate_work_root(work_root)
    _validate_ffmpeg(ffmpeg)
    payload = _validate_mp4_bytes(mp4_bytes)
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)

    with TemporaryDirectory(dir=work_root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = temporary_root / "input.mp4"
        input_path.write_bytes(payload)

        wav_path = temporary_root / "audio.wav"
        audio_stderr = _run_ffmpeg(
            ffmpeg,
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ],
            stage="audio",
            timeout_seconds=timeout_seconds,
        )
        if not _is_nonempty_regular_file(wav_path):
            raise _missing_output_error("audio output is missing or empty", audio_stderr)

        frame_dir = temporary_root / "frames"
        frame_dir.mkdir()
        frame_stderr = _run_ffmpeg(
            ffmpeg,
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                "fps=10",
                str(frame_dir / "frame_%06d.png"),
            ],
            stage="frame",
            timeout_seconds=timeout_seconds,
        )
        if not _has_nonempty_png(frame_dir):
            raise _missing_output_error(
                "frame output contains no nonempty PNG files", frame_stderr
            )

        yield DecodedMedia(wav_path=wav_path, frame_dir=frame_dir)


def _validate_work_root(work_root: object) -> None:
    if not isinstance(work_root, Path) or not work_root.is_dir():
        raise MediaError(f"work_root must be an existing directory: {work_root}")


def _validate_ffmpeg(ffmpeg: object) -> None:
    if (
        not isinstance(ffmpeg, Path)
        or not ffmpeg.is_file()
        or not os.access(ffmpeg, os.X_OK)
    ):
        raise MediaError(f"ffmpeg must be an executable regular file: {ffmpeg}")


def _validate_mp4_bytes(value: object) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise MediaError("mp4_bytes must be nonempty bytes-like data")
    payload = bytes(value)
    if not payload:
        raise MediaError("mp4_bytes must be nonempty bytes-like data")
    return payload


def _validate_timeout_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaError("timeout_seconds must be a finite positive number")
    timeout_seconds = float(value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise MediaError("timeout_seconds must be a finite positive number")
    return timeout_seconds


def _run_ffmpeg(
    ffmpeg: Path, arguments: list[str], *, stage: str, timeout_seconds: float
) -> str:
    try:
        completed = subprocess.run(
            [str(ffmpeg), *arguments],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise MediaError(
            f"{stage} decoding timed out after {timeout_seconds:g} seconds"
        ) from error
    except OSError as error:
        raise MediaError(f"{stage} ffmpeg invocation failed: {error}") from error

    stderr = completed.stderr or ""
    if completed.returncode != 0:
        detail = stderr.strip() or (
            f"ffmpeg exited with status {completed.returncode} without stderr"
        )
        raise MediaError(f"{stage} decode failed: {detail}")
    return stderr


def _missing_output_error(message: str, stderr: str) -> MediaError:
    detail = stderr.strip()
    if detail:
        return MediaError(f"{message} after ffmpeg succeeded: {detail}")
    return MediaError(f"{message} after ffmpeg succeeded")


def _is_nonempty_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _has_nonempty_png(frame_dir: Path) -> bool:
    try:
        return any(
            _is_nonempty_regular_file(candidate) for candidate in frame_dir.glob("*.png")
        )
    except OSError:
        return False
