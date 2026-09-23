"""Small ISO base media file format readers used by the raw-video audit."""

from __future__ import annotations

import struct


_CONTAINER_TYPES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"meta"}


def parse_mvhd_duration(payload: bytes) -> float:
    """Return the movie duration in seconds from an MP4 ``mvhd`` atom."""

    for box_type, box_payload in _walk_boxes(payload):
        if box_type != b"mvhd":
            continue
        if len(box_payload) < 4:
            break
        version = box_payload[0]
        if version == 0 and len(box_payload) >= 20:
            timescale, duration = struct.unpack_from(">II", box_payload, 12)
        elif version == 1 and len(box_payload) >= 32:
            timescale = struct.unpack_from(">I", box_payload, 20)[0]
            duration = struct.unpack_from(">Q", box_payload, 24)[0]
        else:
            raise ValueError("unsupported or truncated mvhd atom")
        if timescale == 0:
            raise ValueError("mvhd timescale is zero")
        return duration / timescale
    raise ValueError("MP4 does not contain an mvhd atom")


def _walk_boxes(data: bytes) -> list[tuple[bytes, bytes]]:
    boxes: list[tuple[bytes, bytes]] = []
    offset = 0
    while offset + 8 <= len(data):
        size, box_type = struct.unpack_from(">I4s", data, offset)
        header_size = 8
        if size == 1:
            if offset + 16 > len(data):
                raise ValueError("truncated extended-size MP4 atom")
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = len(data) - offset
        if size < header_size or offset + size > len(data):
            raise ValueError("invalid MP4 atom size")
        box_payload = data[offset + header_size : offset + size]
        boxes.append((box_type, box_payload))
        if box_type in _CONTAINER_TYPES:
            boxes.extend(_walk_boxes(box_payload[4:] if box_type == b"meta" and len(box_payload) >= 4 else box_payload))
        offset += size
    return boxes
