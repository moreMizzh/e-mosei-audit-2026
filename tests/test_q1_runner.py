from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from e_mosei_audit.archive import ArchiveMember
from e_mosei_audit.q1.config import Q1Config
from e_mosei_audit.q1.extractors import WordInterval
from e_mosei_audit.q1.media import DecodedMedia


@dataclass
class _Archive:
    members: list[ArchiveMember]
    content: dict[str, bytes]
    failed_member: str | None = None
    verify_count: int = 0
    list_count: int = 0
    read_calls: list[str] | None = None

    def __post_init__(self) -> None:
        self.read_calls = []

    def verify(self) -> None:
        self.verify_count += 1

    def list_members(self) -> list[ArchiveMember]:
        self.list_count += 1
        return self.members

    def read_bytes(self, member_path: str) -> bytes:
        assert self.read_calls is not None
        self.read_calls.append(member_path)
        if member_path == self.failed_member:
            raise OSError("fixture archive read failure")
        return self.content[member_path]


class _Aligner:
    def align(self, wav_path: Path, transcript: str, duration: float) -> tuple[WordInterval, ...]:
        return (WordInterval(transcript, 0.0, duration, 0, len(transcript)),)


class _Text:
    def encode(
        self, transcript: str, words: tuple[WordInterval, ...]
    ) -> tuple[np.ndarray, tuple[WordInterval, ...]]:
        return np.full((1, 768), 2.0, dtype=np.float32), words


class _Audio:
    def extract(self, wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.full((1, 25), 3.0, dtype=np.float32),
            np.asarray([0.0]),
            np.asarray([1.0]),
        )


class _Vision:
    def __init__(self) -> None:
        self.close_count = 0

    def extract(self, frame_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.zeros((1, 56), dtype=np.float32)
        values[0, -1] = 1.0
        return values, np.asarray([0.0]), np.asarray([1.0])

    def close(self) -> None:
        self.close_count += 1


def _config(tmp_path: Path) -> Q1Config:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"archive")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ]; then\n"
        "  echo 'fake-ffmpeg version 1.0'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o755)
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    return Q1Config(
        audit_dir=audit_dir,
        archive=archive,
        seven_zip=seven_zip,
        ffmpeg=ffmpeg,
        model_cache=model_cache,
        output_dir=tmp_path / "q1-output",
    )


def _write_audit_rows(audit_dir: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "sample_id",
        "video_id",
        "clip_id",
        "member_path",
        "text",
        "mapping_status",
        "duration_seconds",
        "duration_status",
    )
    with (audit_dir / "raw_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(sample_id: str, *, member_path: str | None = None, **updates: object) -> dict[str, object]:
    video_id, clip_id = sample_id.rsplit("_", maxsplit=1)
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "clip_id": clip_id,
        "member_path": member_path or f"root/{video_id}/{clip_id}.mp4",
        "text": "hello",
        "mapping_status": "matched",
        "duration_seconds": "1.0",
        "duration_status": "parsed",
        **updates,
    }


def _archive_for(rows: list[dict[str, object]], *, failed_member: str | None = None) -> _Archive:
    members = [
        ArchiveMember(path=str(row["member_path"]), size=1, packed_size=1, is_directory=False)
        for row in rows
    ]
    return _Archive(
        members=members,
        content={str(row["member_path"]): b"mp4" for row in rows},
        failed_member=failed_member,
    )


@pytest.fixture
def fake_decode(monkeypatch, tmp_path: Path) -> None:
    from e_mosei_audit.q1 import runner

    @contextmanager
    def decode(ffmpeg: Path, mp4_bytes: bytes, work_root: Path):
        assert mp4_bytes == b"mp4"
        assert work_root.is_dir()
        media_root = tmp_path / "decoded"
        media_root.mkdir(exist_ok=True)
        wav_path = media_root / "audio.wav"
        wav_path.write_bytes(b"wav")
        frame_dir = media_root / "frames"
        frame_dir.mkdir(exist_ok=True)
        for index in range(4):
            (frame_dir / f"frame_{index:06d}.png").write_bytes(b"png")
        yield DecodedMedia(wav_path=wav_path, frame_dir=frame_dir)

    monkeypatch.setattr(runner, "decode_member", decode)


def _extractors():
    from e_mosei_audit.q1.runner import Q1Extractors

    return Q1Extractors(
        aligner=_Aligner(), text=_Text(), audio=_Audio(), vision=_Vision()
    )


def test_run_q1_writes_success_and_failure_coverage_with_evidence(
    fake_decode, tmp_path: Path
) -> None:
    from e_mosei_audit.q1.runner import run_q1

    config = _config(tmp_path)
    rows = [_row("video-a_01"), _row("video-b_01")]
    _write_audit_rows(config.audit_dir, rows)
    failed_member = str(rows[1]["member_path"])
    archive = _archive_for(rows, failed_member=failed_member)

    extractors = _extractors()
    summary = run_q1(config, archive=archive, extractors=extractors, limit=2)

    with (config.output_dir / "q1_samples.csv").open(encoding="utf-8", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    evidence = json.loads(
        (config.output_dir / "alignment_evidence" / "video-a_01.json").read_text(
            encoding="utf-8"
        )
    )
    feature_contract = json.loads(
        (config.output_dir / "feature_contract.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (config.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    typical = (config.output_dir / "typical_sample.md").read_text(encoding="utf-8")
    assert summary == {"coverage_count": 2, "success_count": 1, "failed_count": 1}
    assert [row["status"] for row in coverage] == ["success", "failed"]
    assert coverage[0]["feature_path"] == "features/video-a_01.npz"
    assert coverage[0]["evidence_path"] == "alignment_evidence/video-a_01.json"
    assert coverage[1]["failure_reason"] == "fixture archive read failure"
    assert (config.output_dir / coverage[0]["feature_path"]).is_file()
    assert evidence["transcript"] == "hello"
    assert evidence["slots"][0] == [0.0, 0.02]
    assert evidence["masks"]["vision"][0] is True
    assert evidence["face_presence"]["name"] == "face_presence"
    assert evidence["face_presence"]["meaning"] == (
        "1.0 denotes a detected face, not a detector confidence; frames without a "
        "detected face produce no visual row."
    )
    assert evidence["original_words"][0]["word"] == "hello"
    assert evidence["original_words"][0]["overlap_slot_indexes"] == list(range(50))
    assert evidence["token_intervals"][0] == {
        "text": "hello",
        "char_start": 0,
        "char_end": 5,
        "start": 0.0,
        "end": 1.0,
        "overlap_slot_indexes": list(range(50)),
    }
    assert evidence["audio_windows"][0] == {
        "start": 0.0,
        "end": 1.0,
        "overlap_slot_indexes": list(range(50)),
    }
    assert evidence["detected_visual_rows"][0] == {
        "start": 0.0,
        "end": 1.0,
        "frame_index": 0,
        "overlap_slot_indexes": list(range(50)),
    }
    assert evidence["decoded_frame_count"] == 4
    assert evidence["detected_face_frame_count"] == 1
    assert evidence["face_detection_rate"] == 0.25
    assert "values" not in evidence["audio_windows"][0]
    assert "values" not in evidence["detected_visual_rows"][0]
    assert manifest["reproducibility"]["source_archive"]["zip"]["size"] == 7
    assert len(manifest["reproducibility"]["source_archive"]["zip"]["sha256"]) == 64
    assert manifest["reproducibility"]["tool_versions"]["ffmpeg"]["version"] == (
        "fake-ffmpeg version 1.0"
    )
    assert manifest["reproducibility"]["parameters"]["slot_count"] == 50
    assert manifest["reproducibility"]["models"]["bert"]["id"] == "bert-base-uncased"
    assert manifest["reproducibility"]["models"]["nltk_punkt_tab"] == {
        "id": "tokenizers/punkt_tab/english.pickle",
        "local_cache": str(config.model_cache / "nltk_data"),
    }
    assert isinstance(manifest["reproducibility"]["packages"]["nltk"], str)
    assert feature_contract["reproducibility"]["parameters"]["vision"]["fps"] == 10
    assert feature_contract["output_counts"] == summary
    table_rows = [
        line
        for line in typical.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    assert len(table_rows) == 50
    assert "| 0 | 0.000000 | 0.020000 | hello | 0 |" in typical
    assert archive.verify_count == 1
    assert archive.list_count == 1
    assert archive.read_calls == [str(row["member_path"]) for row in rows]
    assert extractors.vision.close_count == 1


def test_word_evidence_records_alignment_normalization_only_when_used() -> None:
    from e_mosei_audit.q1 import runner

    records = runner._text_interval_records(
        "In 2008,",
        (
            WordInterval("In", 0.0, 0.1, 0, 2),
            WordInterval("2008,", 0.1, 0.6, 3, 8, "two thousand eight"),
        ),
        np.asarray([[0.0, 1.0]]),
        key="word",
    )

    assert "aligned_text" not in records[0]
    assert records[1]["aligned_text"] == "two thousand eight"


def test_run_q1_marks_noneligible_audit_row_failed_without_reading_archive(tmp_path: Path) -> None:
    from e_mosei_audit.q1.runner import run_q1

    config = _config(tmp_path)
    rows = [_row("video-a_01", mapping_status="missing_video")]
    _write_audit_rows(config.audit_dir, rows)
    archive = _archive_for(rows)

    summary = run_q1(config, archive=archive, extractors=_extractors(), limit=1)

    with (config.output_dir / "q1_samples.csv").open(encoding="utf-8", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    assert summary == {"coverage_count": 1, "success_count": 0, "failed_count": 1}
    assert coverage[0]["failure_reason"] == "mapping_status must be matched"
    assert archive.read_calls == []
    assert "No successful sample was extracted." in (
        config.output_dir / "typical_sample.md"
    ).read_text(encoding="utf-8")


def test_run_q1_default_missing_punkt_failure_does_not_create_output_or_read_members(
    monkeypatch, tmp_path: Path
) -> None:
    from e_mosei_audit.q1 import runner

    config = _config(tmp_path)
    rows = [_row("video-a_01")]
    _write_audit_rows(config.audit_dir, rows)
    archive = _archive_for(rows)
    monkeypatch.setattr(
        runner,
        "build_whisperx_aligner",
        lambda model_cache: (_ for _ in ()).throw(
            RuntimeError(
                f"punkt_tab missing in model_cache {model_cache}; downloads are disabled"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="punkt_tab.*downloads are disabled"):
        runner.run_q1(config, archive=archive, limit=1)

    assert not config.output_dir.exists()
    assert archive.read_calls == []


def test_run_q1_ffmpeg_preflight_failure_leaves_output_absent_without_archive_reads(
    tmp_path: Path,
) -> None:
    from e_mosei_audit.q1.runner import run_q1

    config = _config(tmp_path)
    config.ffmpeg.chmod(0o644)
    rows = [_row("video-a_01")]
    _write_audit_rows(config.audit_dir, rows)
    archive = _archive_for(rows)

    with pytest.raises(RuntimeError, match="ffmpeg preflight"):
        run_q1(config, archive=archive, extractors=_extractors(), limit=1)

    assert not config.output_dir.exists()
    assert archive.verify_count == 0
    assert archive.list_count == 0
    assert archive.read_calls == []


def test_run_q1_limit_writes_only_requested_coverage_rows(fake_decode, tmp_path: Path) -> None:
    from e_mosei_audit.q1.runner import run_q1

    config = _config(tmp_path)
    rows = [_row("video-a_01"), _row("video-b_01"), _row("video-c_01")]
    _write_audit_rows(config.audit_dir, rows)
    archive = _archive_for(rows)

    summary = run_q1(config, archive=archive, extractors=_extractors(), limit=1)

    with (config.output_dir / "q1_samples.csv").open(encoding="utf-8", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    assert summary == {"coverage_count": 1, "success_count": 1, "failed_count": 0}
    assert [row["sample_id"] for row in coverage] == ["video-a_01"]
    assert archive.read_calls == [str(rows[0]["member_path"])]


def test_run_q1_rejects_existing_output_before_archive_or_dependency_work(tmp_path: Path) -> None:
    from e_mosei_audit.q1.runner import run_q1

    config = _config(tmp_path)
    _write_audit_rows(config.audit_dir, [_row("video-a_01")])
    config.output_dir.mkdir()
    archive = _archive_for([])

    with pytest.raises(FileExistsError, match="already exists"):
        run_q1(config, archive=archive, extractors=_extractors(), limit=1)

    assert archive.verify_count == 0
    assert archive.list_count == 0
