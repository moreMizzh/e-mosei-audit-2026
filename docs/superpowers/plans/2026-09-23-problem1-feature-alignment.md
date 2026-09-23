# 问题 1：原始多模态特征与时序对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only, reproducible pipeline that turns the 100 audited Attachment 1 videos into three aligned 50-slot feature sequences with evidence and coverage records.

**Architecture:** The pipeline reads only successful rows from the existing audit output and streams one MP4 at a time from `SevenZipArchive` into a temporary workspace. Pure NumPy modules construct physical time slots and serialize the contract; optional external adapters decode media, obtain word times, and extract fixed text/audio/vision vectors. A runner writes an evidence-rich output directory only after preflight succeeds, while individual sample failures remain explicit records and produce a nonzero CLI status.

**Tech Stack:** Python 3.12, NumPy, Pandas, 7-Zip, FFmpeg, Hugging Face Transformers (`bert-base-uncased`), WhisperX with `facebook/wav2vec2-base-960h`, OpenSMILE eGeMAPSv02, MediaPipe Face Landmarker, Pillow, pytest.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Declares the optional `q1` runtime group and pytest marker. |
| `src/e_mosei_audit/q1/config.py` | Parses and validates a TOML config and all explicit tool/model paths. |
| `src/e_mosei_audit/q1/timeline.py` | Builds 50 physical time intervals and performs overlap-weighted pooling. |
| `src/e_mosei_audit/q1/contracts.py` | Defines a validated successful sample, stores compressed NPZ data, and writes the 100-row coverage CSV. |
| `src/e_mosei_audit/q1/media.py` | Writes one archive member into a temporary directory and invokes FFmpeg to obtain WAV and 10 FPS PNG frames. |
| `src/e_mosei_audit/q1/extractors.py` | Provides narrow text, word-alignment, audio, and vision adapter interfaces plus concrete configured implementations. |
| `src/e_mosei_audit/q1/runner.py` | Orchestrates preflight, one-video-at-a-time extraction, evidence, manifests, and partial-failure status. |
| `src/e_mosei_audit/cli.py` | Adds the `extract-q1` subcommand without changing `audit`. |
| `tests/test_q1_timeline.py` | Tests time intervals and weighted pooling without multimedia dependencies. |
| `tests/test_q1_config.py` | Tests explicit tool/model/output path validation. |
| `tests/test_q1_contracts.py` | Tests NPZ/CSV/JSON contract validity, output safety, and 100-row coverage. |
| `tests/test_q1_media.py` | Tests FFmpeg command failures and temporary-file boundaries using a fake executable. |
| `tests/test_q1_extractors.py` | Tests adapter-free text pooling and lazy dependency failure messages. |
| `tests/test_q1_runner.py` | Tests runner behavior with fake archive and extractor adapters. |
| `tests/test_q1_real_smoke.py` | Opt-in test that processes one configured real audited video. |
| `docs/q1-config.example.toml` | Provides a copyable explicit local-tool/model configuration. |
| `README.md` | Documents setup, models, smoke/full commands, output semantics, and submission-size check. |

### Task 1: Add the Q1 configuration contract and dependency boundary

**Files:**
- Modify: `pyproject.toml`
- Create: `src/e_mosei_audit/q1/__init__.py`
- Create: `src/e_mosei_audit/q1/config.py`
- Create: `docs/q1-config.example.toml`
- Test: `tests/test_q1_config.py`

- [ ] **Step 1: Write the failing configuration tests.**

```python
from pathlib import Path

import pytest

from e_mosei_audit.q1.config import Q1Config, load_config


def test_load_config_requires_existing_explicit_runtime_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "q1.toml"
    config_path.write_text(
        """[paths]
audit_dir = "audit"
archive = "data.zip"
seven_zip = "7za"
ffmpeg = "ffmpeg"
model_cache = "models"
output_dir = "output"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="audit_dir"):
        load_config(config_path)


def test_load_config_returns_immutable_explicit_paths(tmp_path: Path) -> None:
    for name in ("audit", "models"):
        (tmp_path / name).mkdir()
    for name in ("data.zip", "7za", "ffmpeg"):
        (tmp_path / name).write_bytes(b"")
    config_path = tmp_path / "q1.toml"
    config_path.write_text(
        """[paths]
audit_dir = "audit"
archive = "data.zip"
seven_zip = "7za"
ffmpeg = "ffmpeg"
model_cache = "models"
output_dir = "output"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == Q1Config(
        audit_dir=tmp_path / "audit",
        archive=tmp_path / "data.zip",
        seven_zip=tmp_path / "7za",
        ffmpeg=tmp_path / "ffmpeg",
        model_cache=tmp_path / "models",
        output_dir=tmp_path / "output",
    )
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_config.py -v`  
Expected: FAIL because `e_mosei_audit.q1.config` does not exist.

- [ ] **Step 3: Declare the optional dependency group and implement path validation.**

Add this to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
q1 = [
  "mediapipe>=0.10.14",
  "opensmile>=2.5.1",
  "Pillow>=10.4",
  "torch>=2.4",
  "transformers>=4.45",
  "whisperx>=3.3",
]
```

Implement `config.py` around the following public API. Resolve all relative paths against the TOML file, require directories and regular files to exist, and do not create any runtime paths in this module:

```python
from dataclasses import dataclass
from pathlib import Path
import dataclasses
import tomllib


@dataclass(frozen=True)
class Q1Config:
    audit_dir: Path
    archive: Path
    seven_zip: Path
    ffmpeg: Path
    model_cache: Path
    output_dir: Path


def load_config(path: Path) -> Q1Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    values = raw.get("paths")
    if not isinstance(values, dict):
        raise ValueError("q1 config requires a [paths] table")
    base = path.parent
    resolved = {name: (base / str(values[name])).resolve() for name in Q1Config.__dataclass_fields__}
    for name in ("audit_dir", "model_cache"):
        if not resolved[name].is_dir():
            raise ValueError(f"{name} must be an existing directory: {resolved[name]}")
    for name in ("archive", "seven_zip", "ffmpeg"):
        if not resolved[name].is_file():
            raise ValueError(f"{name} must be an existing file: {resolved[name]}")
    if not resolved["output_dir"].parent.is_dir():
        raise ValueError(f"output_dir parent must exist: {resolved['output_dir'].parent}")
    return Q1Config(**resolved)
```

Create the example config with relative entries matching the audit workflow:

```toml
# Copy this file to the repository root as q1.toml before running the CLI.
[paths]
audit_dir = "artifacts/data-audit-v2"
archive = "E题数据 (2).zip"
seven_zip = ".tools/bin/7za"
ffmpeg = ".tools/bin/ffmpeg"
model_cache = ".tools/models"
output_dir = "artifacts/q1-default"
```

- [ ] **Step 4: Run configuration tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_config.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the contract boundary.**

```bash
git add pyproject.toml src/e_mosei_audit/q1/__init__.py src/e_mosei_audit/q1/config.py docs/q1-config.example.toml tests/test_q1_config.py
git commit -m "feat: add problem one runtime configuration"
```

### Task 2: Implement physical time slots and overlap-weighted pooling

**Files:**
- Create: `src/e_mosei_audit/q1/timeline.py`
- Test: `tests/test_q1_timeline.py`

- [ ] **Step 1: Write the failing numerical tests.**

```python
import numpy as np
import pytest

from e_mosei_audit.q1.timeline import make_slots, pool_intervals, pool_moments


def test_make_slots_divides_original_duration_into_fifty_physical_intervals() -> None:
    slots = make_slots(10.0)

    assert slots.shape == (50, 2)
    assert slots[0].tolist() == [0.0, 0.2]
    assert slots[-1].tolist() == [9.8, 10.0]


def test_pool_intervals_uses_overlap_not_interval_midpoints() -> None:
    values = np.array([[2.0], [10.0]], dtype=np.float32)
    starts = np.array([0.0, 0.5])
    ends = np.array([0.5, 1.0])
    slots = np.array([[0.25, 0.75], [0.75, 1.0]])

    pooled, mask = pool_intervals(values, starts, ends, slots)

    assert pooled[:, 0] == pytest.approx([6.0, 10.0])
    assert mask.tolist() == [True, True]


def test_pool_intervals_distinguishes_zero_features_from_empty_slots() -> None:
    values = np.zeros((1, 2), dtype=np.float32)
    pooled, mask = pool_intervals(values, np.array([0.0]), np.array([0.2]), np.array([[0.0, 0.2], [0.2, 0.4]]))

    assert pooled.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert mask.tolist() == [True, False]


def test_pool_moments_concatenates_mean_and_standard_deviation() -> None:
    values = np.array([[2.0], [6.0]], dtype=np.float32)

    pooled, mask = pool_moments(values, np.array([0.0, 0.5]), np.array([0.5, 1.0]), np.array([[0.0, 1.0]]))

    assert pooled.tolist() == [[4.0, 2.0]]
    assert mask.tolist() == [True]
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_timeline.py -v`  
Expected: FAIL because `e_mosei_audit.q1.timeline` does not exist.

- [ ] **Step 3: Implement the pure NumPy API.**

```python
import numpy as np


def make_slots(duration: float, count: int = 50) -> np.ndarray:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if count < 1:
        raise ValueError("count must be positive")
    return np.column_stack((np.linspace(0.0, duration, count, endpoint=False), np.linspace(duration / count, duration, count)))


def pool_intervals(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray, slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or starts.shape != ends.shape or starts.ndim != 1 or len(starts) != len(values):
        raise ValueError("values and intervals must have matching leading dimensions")
    result = np.zeros((len(slots), values.shape[1]), dtype=np.float32)
    mask = np.zeros(len(slots), dtype=bool)
    for index, (slot_start, slot_end) in enumerate(slots):
        overlap = np.maximum(0.0, np.minimum(ends, slot_end) - np.maximum(starts, slot_start))
        weight = overlap.sum()
        if weight:
            result[index] = (values * overlap[:, None]).sum(axis=0) / weight
            mask[index] = True
    return result, mask


def pool_moments(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray, slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means, mask = pool_intervals(values, starts, ends, slots)
    result = np.zeros((len(slots), values.shape[1] * 2), dtype=np.float32)
    result[:, : values.shape[1]] = means
    for index, (slot_start, slot_end) in enumerate(slots):
        overlap = np.maximum(0.0, np.minimum(ends, slot_end) - np.maximum(starts, slot_start))
        weight = overlap.sum()
        if weight:
            variance = ((values - means[index]) ** 2 * overlap[:, None]).sum(axis=0) / weight
            result[index, values.shape[1] :] = np.sqrt(variance)
    return result, mask
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_timeline.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the alignment primitive.**

```bash
git add src/e_mosei_audit/q1/timeline.py tests/test_q1_timeline.py
git commit -m "feat: add physical time slot pooling"
```

### Task 3: Build output contracts and complete 100-row coverage accounting

**Files:**
- Create: `src/e_mosei_audit/q1/contracts.py`
- Test: `tests/test_q1_contracts.py`

- [ ] **Step 1: Write the failing contract tests.**

```python
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from e_mosei_audit.q1.contracts import SampleFeatures, write_coverage


def test_sample_features_rejects_a_wrong_text_shape(tmp_path: Path) -> None:
    sample = SampleFeatures("v_01", np.zeros((49, 768)), np.zeros((50, 50)), np.zeros((50, 56)), np.ones((50, 2)))

    with pytest.raises(ValueError, match="text"):
        sample.write(tmp_path / "v_01.npz")


def test_successful_sample_writes_float16_arrays_and_independent_masks(tmp_path: Path) -> None:
    sample = SampleFeatures(
        "v_01",
        np.ones((50, 768)),
        np.zeros((50, 50)),
        np.ones((50, 56)),
        np.column_stack((np.arange(50), np.arange(1, 51))),
        text_mask=np.ones(50, dtype=bool),
        audio_mask=np.ones(50, dtype=bool),
        vision_mask=np.zeros(50, dtype=bool),
    )
    path = tmp_path / "v_01.npz"

    sample.write(path)

    stored = np.load(path)
    assert stored["audio"].dtype == np.float16
    assert stored["audio_mask"].tolist() == [True] * 50
    assert stored["vision_mask"].tolist() == [False] * 50


def test_write_coverage_keeps_failure_rows_and_requires_one_record_per_input(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "a_01", "status": "success", "feature_path": "features/a_01.npz", "failure_reason": ""},
        {"sample_id": "b_01", "status": "failed", "feature_path": "", "failure_reason": "ffmpeg failed"},
    ]

    write_coverage(tmp_path / "q1_samples.csv", rows, expected_count=2)

    with (tmp_path / "q1_samples.csv").open(encoding="utf-8", newline="") as stream:
        assert [row["status"] for row in csv.DictReader(stream)] == ["success", "failed"]
    assert json.loads((tmp_path / "q1_summary.json").read_text(encoding="utf-8"))["failed_count"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_contracts.py -v`  
Expected: FAIL because `e_mosei_audit.q1.contracts` does not exist.

- [ ] **Step 3: Implement shape validation and deterministic output writers.**

Implement `SampleFeatures.write()` to require exact feature shapes `(50, 768)`, `(50, 50)`, `(50, 56)`, a `(50, 2)` monotonic timestamp matrix, and three boolean masks of length 50. It must create only the requested parent directory, use `np.savez_compressed`, and cast only the three feature arrays to `float16`.

Implement `write_coverage(path, rows, expected_count)` with the fixed columns below. It must reject duplicate sample IDs, reject an unexpected row count, atomically write `q1_samples.csv`, and write `q1_summary.json` next to it.

```python
_COVERAGE_COLUMNS = (
    "sample_id", "video_id", "clip_id", "member_path", "duration_seconds",
    "status", "feature_path", "evidence_path", "text_shape", "audio_shape",
    "vision_shape", "failure_reason",
)
```

- [ ] **Step 4: Run the contract tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_contracts.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit contract storage.**

```bash
git add src/e_mosei_audit/q1/contracts.py tests/test_q1_contracts.py
git commit -m "feat: add problem one feature contracts"
```

### Task 4: Add one-member media decoding with explicit FFmpeg failures

**Files:**
- Create: `src/e_mosei_audit/q1/media.py`
- Test: `tests/test_q1_media.py`

- [ ] **Step 1: Write the failing FFmpeg boundary tests.**

```python
from pathlib import Path

import pytest

from e_mosei_audit.q1.media import MediaError, decode_member


def test_decode_member_reports_ffmpeg_stderr_without_leaving_media_files(tmp_path: Path) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\necho decode-failed >&2\nexit 7\n", encoding="utf-8")
    ffmpeg.chmod(0o755)

    with pytest.raises(MediaError, match="decode-failed"):
        with decode_member(ffmpeg, b"not-mp4", tmp_path) as media:
            raise AssertionError(media)

    assert list(tmp_path.iterdir()) == [ffmpeg]
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_media.py -v`  
Expected: FAIL because `e_mosei_audit.q1.media` does not exist.

- [ ] **Step 3: Implement the temporary media context manager.**

Expose this concrete API:

```python
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedMedia:
    wav_path: Path
    frame_dir: Path


@contextmanager
def decode_member(ffmpeg: Path, mp4_bytes: bytes, work_root: Path) -> Iterator[DecodedMedia]:
    ...
```

Write `mp4_bytes` only to a `TemporaryDirectory(dir=work_root)`. Invoke FFmpeg twice with `check=False`: once to create a mono 16 kHz WAV, and once with `-vf fps=10` to create `frame_%06d.png`. On either failure, raise `MediaError` containing decoded stderr; always clean the temporary directory. Do not invoke `ffprobe` and do not extract any other archive member.

- [ ] **Step 4: Run media tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_media.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the decoder boundary.**

```bash
git add src/e_mosei_audit/q1/media.py tests/test_q1_media.py
git commit -m "feat: decode one problem one video at a time"
```

### Task 5: Implement explicit external extractor adapters

**Files:**
- Create: `src/e_mosei_audit/q1/extractors.py`
- Test: `tests/test_q1_extractors.py`

- [ ] **Step 1: Write failing tests against dependency-free fake adapters.**

```python
import numpy as np

from e_mosei_audit.q1.extractors import WordInterval, text_slot_features


def test_text_slot_features_pools_only_tokens_with_word_time_evidence() -> None:
    embeddings = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    intervals = [WordInterval("hello", 0.0, 0.4, 0, 5), WordInterval("world", 0.6, 1.0, 6, 11)]
    slots = np.array([[0.0, 0.5], [0.5, 1.0]])

    values, mask = text_slot_features(embeddings, intervals, slots)

    assert values.tolist() == [[1.0, 3.0], [5.0, 7.0]]
    assert mask.tolist() == [True, True]
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_extractors.py -v`  
Expected: FAIL because `e_mosei_audit.q1.extractors` does not exist.

- [ ] **Step 3: Implement interfaces before concrete imports.**

Define `WordInterval(word, start, end, char_start, char_end)`, `WordAligner.align(wav_path, transcript, duration)`, `TextEncoder.encode(transcript, words)`, `AudioExtractor.extract(wav_path)`, and `VisionExtractor.extract(frame_dir)`. Each extractor returns vectors plus physical `starts` and `ends`; no adapter returns pre-pooled slots. Use `pool_intervals` to perform all pooling centrally.

`text_slot_features` must call `pool_intervals` on token embeddings and word intervals and require a one-to-one vector/interval count. It must never use the raw embedding value as a missingness indicator. The runner must use `pool_moments` for the 25-dimensional OpenSMILE windows so its audio output has exactly 50 dimensions.

- [ ] **Step 4: Add concrete adapters with lazy imports and fixed output shapes.**

Implement these constructors:

```python
def build_whisperx_aligner(model_cache: Path) -> WordAligner: ...
def build_bert_encoder(model_cache: Path) -> TextEncoder: ...
def build_opensmile_extractor() -> AudioExtractor: ...
def build_mediapipe_extractor(model_cache: Path) -> VisionExtractor: ...
```

Each constructor imports its third-party package inside the function and raises `RuntimeError` naming the missing package or model asset. The BERT adapter must return 768 dimensions. The OpenSMILE adapter must request `LowLevelDescriptors`, return 25-dimensional windows with timestamps, and leave 50-dimensional mean/std pooling to the runner. The MediaPipe adapter must pad/truncate blendshape scores to 52 and append face area, face-centre x/y, and detection score to exactly 56 dimensions. Empty valid detector output is represented by an empty `(0, 56)` frame array and not a zero-filled frame.

- [ ] **Step 5: Run extractor tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_extractors.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit extractor interfaces.**

```bash
git add src/e_mosei_audit/q1/extractors.py tests/test_q1_extractors.py
git commit -m "feat: add problem one extractor adapters"
```

### Task 6: Orchestrate audit-backed extraction, evidence, and CLI status

**Files:**
- Create: `src/e_mosei_audit/q1/runner.py`
- Modify: `src/e_mosei_audit/cli.py`
- Test: `tests/test_q1_runner.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write a failing runner test with fake dependencies.**

```python
import csv
import json
from pathlib import Path

from e_mosei_audit.q1.runner import run_q1


def test_run_q1_preserves_all_audited_rows_when_one_member_fails(tmp_path: Path, fake_q1_dependencies) -> None:
    audit_dir, archive, config = fake_q1_dependencies(tmp_path, failed_member="root/video-b.mp4")

    summary = run_q1(config, archive=archive)

    with (config.output_dir / "q1_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["success", "failed"]
    assert summary["failed_count"] == 1
    assert json.loads((config.output_dir / "alignment_evidence/video-a_01.json").read_text(encoding="utf-8"))["slots"][0] == [0.0, 0.02]
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_runner.py -v`  
Expected: FAIL because `e_mosei_audit.q1.runner` does not exist.

- [ ] **Step 3: Implement runner invariants.**

`run_q1(config, archive=None, limit=None)` must:

1. Refuse an existing `config.output_dir` before reading any archive member.
2. Load `raw_samples.csv`, require exactly 100 rows, and preserve its sorted order. When `limit` is absent, it must write all 100 coverage rows; when `limit=N`, it may process and write only the first `N` rows for an explicit smoke run.
3. Accept only rows with `mapping_status == "matched"` and `duration_status == "parsed"`; every other row becomes a failed coverage row without attempting extraction.
4. Verify and list the archive once. For each eligible row, call `archive.read_bytes(member_path)`, use `decode_member` in a temporary work root, and invoke the four adapters.
5. Build 50 slots from the audited duration, pool the three modalities, save a `SampleFeatures` NPZ and an evidence JSON containing `sample_id`, original transcript, 50 slot boundaries, words, masks, audio window count, and visual frame count.
6. Catch expected per-sample `MediaError`, `RuntimeError`, `ValueError`, and `OSError`, write a `failed` row with the exception message, then continue.
7. Write `feature_contract.json`, `run_manifest.json`, `run.log`, `q1_samples.csv`, and `typical_sample.md` after the loop. Pick the first successful sample for the typical report. Return counts; `main` must exit `1` when `failed_count > 0` and `0` otherwise.

Add to `cli.py`:

```python
extract = commands.add_parser("extract-q1", help="extract and align Attachment 1 features")
extract.add_argument("--config", required=True, help="TOML path with explicit local tool and model paths")
extract.add_argument("--output", required=True, help="new directory for Q1 derived artifacts")
extract.add_argument("--limit", type=int, help="process only the first N audited rows for a smoke run")
```

The CLI must apply `--output` with `dataclasses.replace(config, output_dir=Path(arguments.output))`, print sorted JSON summary, and preserve `audit` command behavior and exit codes. Add this method to `Q1Config` for test readability:

```python
def with_output_dir(self, output_dir: Path) -> "Q1Config":
    return dataclasses.replace(self, output_dir=output_dir)
```

- [ ] **Step 4: Run runner and CLI tests to verify they pass.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_runner.py tests/test_cli.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the workflow.**

```bash
git add src/e_mosei_audit/q1/runner.py src/e_mosei_audit/cli.py tests/test_q1_runner.py tests/test_cli.py
git commit -m "feat: run problem one feature extraction"
```

### Task 7: Document runtime setup and execute controlled verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_q1_real_smoke.py`

- [ ] **Step 1: Add the opt-in real smoke test.**

```python
import os
from pathlib import Path

import pytest

from e_mosei_audit.q1.config import load_config
from e_mosei_audit.q1.runner import run_q1


@pytest.mark.integration
def test_real_q1_smoke(tmp_path: Path) -> None:
    config_path = os.environ.get("E_MOSEI_Q1_CONFIG")
    if not config_path:
        pytest.skip("set E_MOSEI_Q1_CONFIG to run the real Q1 smoke test")
    config = load_config(Path(config_path))
    config = config.with_output_dir(tmp_path / "q1-smoke")

    summary = run_q1(config, limit=1)

    assert summary["coverage_count"] == 1
    assert summary["success_count"] == 1
```

- [ ] **Step 2: Run the test to verify it is skipped without an explicit config.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_real_smoke.py -v`  
Expected: SKIPPED with the `E_MOSEI_Q1_CONFIG` message.

- [ ] **Step 3: Document provisioning and commands.**

Add a `问题 1 特征提取` section to `README.md` with these exact conditions:

```bash
python3 -m pip install --target .tools/python '.[q1]'
export PYTHONPATH="$PWD/src:$PWD/.tools/python"

# First run only after placing FFmpeg and approved model assets in paths referenced by q1.toml.
python3 -m e_mosei_audit.cli extract-q1 \
  --config q1.toml \
  --output artifacts/q1-smoke \
  --limit 1

# Run only after reviewing the smoke evidence and output-size summary.
python3 -m e_mosei_audit.cli extract-q1 \
  --config q1.toml \
  --output artifacts/q1-full
```

State that models/caches/intermediate media are ignored, that failed samples produce a coverage row and nonzero status, and that `q1_samples.csv` must contain 100 records in a full run.

- [ ] **Step 4: Run the whole dependency-free suite.**

Run: `PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest -v`  
Expected: all existing audit tests and all new pure/fake Q1 tests PASS; the real smoke test SKIPS unless explicitly configured.

- [ ] **Step 5: Run the real smoke test after dependencies and assets are present.**

Run: `E_MOSEI_Q1_CONFIG="$PWD/q1.toml" PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest tests/test_q1_real_smoke.py -v`  
Expected: PASS; inspect `q1_samples.csv`, one NPZ, and `typical_sample.md` before authorizing the full 100-sample run.

- [ ] **Step 6: Commit documentation and smoke coverage.**

```bash
git add README.md tests/test_q1_real_smoke.py
git commit -m "docs: add problem one extraction guide"
```

## Plan self-review

- **Spec coverage:** Task 1 covers explicit paths and no silent runtime setup; Task 2 implements physical 50-slot alignment; Task 3 covers compact tensors, masks, contracts, and 100-row coverage; Task 4 enforces one-video-at-a-time read-only decoding; Task 5 fixes every specified model and dimensional interface; Task 6 writes all required evidence, manifests, errors, and CLI behavior; Task 7 covers smoke-first verification and documentation.
- **No omissions:** The plan specifies all paths, commands, public APIs, output shapes, error cases, and commit boundaries. External model assets remain deliberately explicit inputs rather than implicit downloads.
- **Consistency:** All later tasks use the fixed `(50, 768)`, `(50, 50)`, `(50, 56)` contract and the same `Q1Config`, 50 physical slots, coverage CSV, and output-first failure semantics.
