from __future__ import annotations

import struct

import pytest

from e_mosei_audit.media import parse_mvhd_duration


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _mp4_with_mvhd(version: int, timescale: int, duration: int) -> bytes:
    if version == 0:
        payload = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, timescale, duration)
    else:
        payload = bytes([1, 0, 0, 0]) + struct.pack(">QQIQ", 0, 0, timescale, duration)
    return _box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _box(b"moov", _box(b"mvhd", payload))


def test_parses_version_zero_movie_header_duration() -> None:
    assert parse_mvhd_duration(_mp4_with_mvhd(0, 1_000, 2_648)) == pytest.approx(2.648)


def test_parses_version_one_movie_header_duration() -> None:
    assert parse_mvhd_duration(_mp4_with_mvhd(1, 48_000, 1_658_400)) == pytest.approx(34.55)


def test_rejects_mp4_without_movie_header() -> None:
    with pytest.raises(ValueError, match="mvhd"):
        parse_mvhd_duration(_box(b"ftyp", b"isom"))
