from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def _media() -> object:
    spec = importlib.util.find_spec("e_mosei_audit.q1.media")
    assert spec is not None, "q1 media module must exist"
    return importlib.import_module("e_mosei_audit.q1.media")


def _fake_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str = "success"
) -> tuple[Path, Path]:
    ffmpeg = tmp_path / "fake_ffmpeg.py"
    log_path = tmp_path / "ffmpeg_calls.jsonl"
    ffmpeg.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
with Path(os.environ["FAKE_FFMPEG_LOG"]).open("a", encoding="utf-8") as log_file:
    log_file.write(json.dumps(args) + "\\n")

mode = os.environ["FAKE_FFMPEG_MODE"]
output = Path(args[-1])
is_audio = output.suffix == ".wav"
if mode == "audio-fail" and is_audio:
    print("audio decoder broke", file=sys.stderr)
    raise SystemExit(7)
if mode == "frame-fail" and not is_audio:
    print("frame decoder broke", file=sys.stderr)
    raise SystemExit(8)
if mode == "missing-audio" and is_audio:
    raise SystemExit(0)
if mode == "missing-frames" and not is_audio:
    raise SystemExit(0)
if mode == "missing-audio-stderr" and is_audio:
    print("audio success diagnostic", file=sys.stderr)
    raise SystemExit(0)
if mode == "missing-frames-stderr" and not is_audio:
    print("frame success diagnostic", file=sys.stderr)
    raise SystemExit(0)
if mode == "zero-byte-frames" and not is_audio:
    output.with_name("frame_000001.png").write_bytes(b"")
    raise SystemExit(0)
if mode == "sleep-audio" and is_audio:
    time.sleep(float(os.environ["FAKE_FFMPEG_SLEEP_SECONDS"]))

if is_audio:
    output.write_bytes(b"RIFF fake wav")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name("frame_000001.png").write_bytes(b"fake png")
"""
    )
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("FAKE_FFMPEG_LOG", str(log_path))
    monkeypatch.setenv("FAKE_FFMPEG_MODE", mode)
    return ffmpeg, log_path


def _calls(log_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_decode_member_decodes_expected_audio_and_frame_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, log_path = _fake_ffmpeg(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with media.decode_member(ffmpeg, memoryview(b"fake mp4"), work_root) as decoded:
        assert isinstance(decoded, media.DecodedMedia)
        assert decoded.wav_path.read_bytes() == b"RIFF fake wav"
        assert (decoded.frame_dir / "frame_000001.png").read_bytes() == b"fake png"

    assert list(work_root.iterdir()) == []
    audio_call, frame_call = _calls(log_path)
    assert audio_call[:5] == ["-hide_banner", "-loglevel", "error", "-y", "-i"]
    assert Path(audio_call[5]).name == "input.mp4"
    assert audio_call[audio_call.index("-ac") + 1] == "1"
    assert audio_call[audio_call.index("-ar") + 1] == "16000"
    assert Path(audio_call[-1]).name == "audio.wav"
    assert frame_call[:5] == ["-hide_banner", "-loglevel", "error", "-y", "-i"]
    assert Path(frame_call[5]).name == "input.mp4"
    assert frame_call[frame_call.index("-vf") + 1] == "fps=10"
    assert Path(frame_call[-1]).parent.name == "frames"
    assert Path(frame_call[-1]).name == "frame_%06d.png"


def test_decode_member_reports_audio_failure_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode="audio-fail")
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match="audio decoder broke"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run when audio decoding fails")

    assert list(work_root.iterdir()) == []


def test_decode_member_reports_frame_failure_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode="frame-fail")
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match="frame decoder broke"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run when frame decoding fails")

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing-audio", "audio output"),
        ("missing-frames", "frame output"),
    ],
)
def test_decode_member_rejects_successful_ffmpeg_without_required_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode=mode)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match=message):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run without both outputs")

    assert list(work_root.iterdir()) == []


def test_decode_member_rejects_zero_byte_png_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode="zero-byte-frames")
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match="frame output"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run with an empty PNG")

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "diagnostic"),
    [
        ("missing-audio-stderr", "audio success diagnostic"),
        ("missing-frames-stderr", "frame success diagnostic"),
    ],
)
def test_decode_member_includes_successful_ffmpeg_stderr_when_output_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, diagnostic: str
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode=mode)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match=diagnostic):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run without required output")

    assert list(work_root.iterdir()) == []


def test_decode_member_times_out_audio_decode_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch, mode="sleep-audio")
    monkeypatch.setenv("FAKE_FFMPEG_SLEEP_SECONDS", "1")
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(
        media.MediaError, match=r"audio decoding timed out after 0\.01 seconds"
    ):
        with media.decode_member(
            ffmpeg, b"fake mp4", work_root, timeout_seconds=0.01
        ):
            pytest.fail("the caller body must not run when decoding times out")

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize(
    "timeout_seconds", [0.0, -1.0, float("nan"), float("inf"), True, "one"]
)
def test_decode_member_rejects_nonpositive_or_nonfinite_timeout_before_temporary_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_seconds: object
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match="timeout_seconds"):
        with media.decode_member(
            ffmpeg, b"fake mp4", work_root, timeout_seconds=timeout_seconds
        ):
            pytest.fail("the caller body must not run with an invalid timeout")

    assert list(work_root.iterdir()) == []


def test_decode_member_preserves_caller_exception_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(LookupError, match="caller failure"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            raise LookupError("caller failure")

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("work_root_name", ["missing-work", "work-file"])
def test_decode_member_rejects_invalid_work_root_before_temporary_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, work_root_name: str
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch)
    work_root = tmp_path / work_root_name
    if work_root_name == "work-file":
        work_root.write_text("not a directory")

    with pytest.raises(media.MediaError, match="work_root"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run with an invalid work root")

    if work_root.is_dir():
        assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["missing", "directory", "not-executable"])
def test_decode_member_rejects_invalid_ffmpeg_before_temporary_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    media = _media()
    work_root = tmp_path / "work"
    work_root.mkdir()
    ffmpeg = tmp_path / "ffmpeg"
    if kind == "directory":
        ffmpeg.mkdir()
    elif kind == "not-executable":
        ffmpeg.write_text("#!/bin/sh\n")
        ffmpeg.chmod(0o644)

    with pytest.raises(media.MediaError, match="ffmpeg"):
        with media.decode_member(ffmpeg, b"fake mp4", work_root):
            pytest.fail("the caller body must not run with an invalid ffmpeg")

    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("mp4_bytes", [b"", bytearray(), memoryview(b""), "not bytes"])
def test_decode_member_rejects_invalid_input_before_temporary_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4_bytes: object
) -> None:
    media = _media()
    ffmpeg, _ = _fake_ffmpeg(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(media.MediaError, match="mp4_bytes"):
        with media.decode_member(ffmpeg, mp4_bytes, work_root):
            pytest.fail("the caller body must not run with invalid input")

    assert list(work_root.iterdir()) == []
