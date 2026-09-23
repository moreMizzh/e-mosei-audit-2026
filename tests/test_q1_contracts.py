from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


COVERAGE_COLUMNS = (
    "sample_id",
    "video_id",
    "clip_id",
    "member_path",
    "duration_seconds",
    "status",
    "feature_path",
    "evidence_path",
    "text_shape",
    "audio_shape",
    "vision_shape",
    "failure_reason",
)


def _contracts() -> ModuleType:
    spec = importlib.util.find_spec("e_mosei_audit.q1.contracts")
    assert spec is not None, "q1 contracts module must exist"
    return importlib.import_module("e_mosei_audit.q1.contracts")


def _valid_sample(sample_id: str = "sample-001") -> object:
    contracts = _contracts()
    slots = np.column_stack(
        (np.arange(50, dtype=np.float64), np.arange(1, 51, dtype=np.float64))
    )
    return contracts.SampleFeatures(
        sample_id=sample_id,
        text=np.zeros((50, 768), dtype=np.float32),
        audio=np.ones((50, 50), dtype=np.float32),
        vision=np.full((50, 56), 2.0, dtype=np.float32),
        slots=slots,
        text_mask=np.resize(np.array([True, False], dtype=bool), 50),
        audio_mask=np.resize(np.array([False, True], dtype=bool), 50),
        vision_mask=np.ones(50, dtype=bool),
    )


def _coverage_row(sample_id: str, *, status: str = "success") -> dict[str, object]:
    values: dict[str, object] = {
        "sample_id": sample_id,
        "video_id": "video-1",
        "clip_id": "clip-1",
        "member_path": "attachment/video-1.mp4",
        "duration_seconds": 12.5,
        "status": status,
        "feature_path": "features/sample-001.npz" if status == "success" else "",
        "evidence_path": "evidence/sample-001.json" if status == "success" else "",
        "text_shape": "(50, 768)" if status == "success" else "",
        "audio_shape": "(50, 50)" if status == "success" else "",
        "vision_shape": "(50, 56)" if status == "success" else "",
        "failure_reason": "" if status == "success" else "missing media stream",
    }
    return {column: values[column] for column in COVERAGE_COLUMNS}


def test_sample_features_write_stores_contract_dtypes_and_preserves_masks(
    tmp_path: Path,
) -> None:
    sample = _valid_sample()
    target = tmp_path / "features" / "sample-001.npz"

    sample.write(target)

    with np.load(target, allow_pickle=False) as stored:
        assert set(stored.files) == {
            "sample_id",
            "text",
            "audio",
            "vision",
            "slots",
            "text_mask",
            "audio_mask",
            "vision_mask",
        }
        assert stored["sample_id"].shape == ()
        assert stored["sample_id"].item() == "sample-001"
        assert stored["text"].dtype == np.float16
        assert stored["audio"].dtype == np.float16
        assert stored["vision"].dtype == np.float16
        assert stored["slots"].dtype == np.float64
        assert stored["text_mask"].dtype == np.bool_
        assert stored["audio_mask"].dtype == np.bool_
        assert stored["vision_mask"].dtype == np.bool_
        assert stored["text"].shape == (50, 768)
        assert stored["audio"].shape == (50, 50)
        assert stored["vision"].shape == (50, 56)
        assert stored["slots"].shape == (50, 2)
        np.testing.assert_array_equal(stored["text"], np.zeros((50, 768)))
        np.testing.assert_array_equal(stored["text_mask"], sample.text_mask)
        np.testing.assert_array_equal(stored["audio_mask"], sample.audio_mask)
        np.testing.assert_array_equal(stored["vision_mask"], sample.vision_mask)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("text", np.zeros((49, 768), dtype=np.float32)),
        ("audio", np.zeros((50, 49), dtype=np.float32)),
        ("vision", np.zeros((50, 55), dtype=np.float32)),
    ],
)
def test_sample_features_write_rejects_wrong_feature_shapes_before_output_creation(
    tmp_path: Path, field: str, invalid: np.ndarray
) -> None:
    sample = replace(_valid_sample(), **{field: invalid})
    target = tmp_path / "not-created" / "sample.npz"

    with pytest.raises(ValueError, match=field):
        sample.write(target)

    assert not target.parent.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("text", np.full((50, 768), np.nan, dtype=np.float32)),
        ("audio", np.full((50, 50), np.inf, dtype=np.float32)),
        ("vision", np.ones((50, 56), dtype=np.complex128)),
    ],
)
def test_sample_features_write_rejects_nonfinite_or_complex_features(
    tmp_path: Path, field: str, invalid: np.ndarray
) -> None:
    sample = replace(_valid_sample(), **{field: invalid})

    with pytest.raises(ValueError, match=field):
        sample.write(tmp_path / "sample.npz")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sample: replace(
            sample,
            slots=np.vstack(
                (
                    sample.slots[:25],
                    sample.slots[25:] + np.array([0.1, 0.0]),
                )
            ),
        ),
        lambda sample: replace(
            sample,
            slots=np.vstack(
                (np.array([[0.0, 0.0]]), sample.slots[1:])
            ),
        ),
        lambda sample: replace(
            sample,
            slots=np.vstack(
                (np.array([[np.nan, 1.0]]), sample.slots[1:])
            ),
        ),
        lambda sample: replace(sample, text_mask=np.zeros(49, dtype=bool)),
        lambda sample: replace(sample, audio_mask=np.zeros(50, dtype=np.int8)),
    ],
    ids=["gapped-slots", "zero-slot", "nonfinite-slot", "short-mask", "nonbool-mask"],
)
def test_sample_features_write_rejects_invalid_slots_or_masks(
    tmp_path: Path, mutate: object
) -> None:
    sample = mutate(_valid_sample())

    with pytest.raises(ValueError):
        sample.write(tmp_path / "sample.npz")


@pytest.mark.parametrize("sample_id", [None, "", "   "])
def test_sample_features_write_rejects_invalid_sample_id(
    tmp_path: Path, sample_id: object
) -> None:
    sample = _valid_sample(sample_id=sample_id)
    target = tmp_path / "not-created" / "sample.npz"

    with pytest.raises(ValueError, match="sample_id"):
        sample.write(target)

    assert not target.parent.exists()


def test_sample_features_write_rejects_non_npz_target_and_duplicate_sample_id(
    tmp_path: Path,
) -> None:
    sample = _valid_sample()

    with pytest.raises(ValueError, match="npz"):
        sample.write(tmp_path / "not-created" / "sample.bin")
    assert not (tmp_path / "not-created").exists()

    sample.write(tmp_path / "first.npz")
    with pytest.raises(ValueError, match="duplicate"):
        sample.write(tmp_path / "second.npz")
    assert not (tmp_path / "second.npz").exists()


@pytest.mark.parametrize(
    ("rows", "expected_count"),
    [
        ([{key: value for key, value in _coverage_row("one").items() if key != "clip_id"}], 1),
        ([{**_coverage_row("one"), "extra": "unexpected"}], 1),
        ([_coverage_row("one"), _coverage_row("one", status="failed")], 2),
        ([_coverage_row("one", status="pending")], 1),
        ([{**_coverage_row("one"), "status": []}], 1),
        ([{**_coverage_row("one"), "feature_path": ""}], 1),
        ([{**_coverage_row("one", status="failed"), "failure_reason": ""}], 1),
        ([_coverage_row("one")], 2),
    ],
    ids=[
        "missing-column",
        "extra-column",
        "duplicate-sample-id",
        "invalid-status",
        "nonstring-status",
        "success-without-feature-path",
        "failed-without-reason",
        "wrong-expected-count",
    ],
)
def test_write_coverage_rejects_duplicate_or_malformed_rows(
    tmp_path: Path, rows: list[dict[str, object]], expected_count: int
) -> None:
    contracts = _contracts()
    output = tmp_path / "q1_samples.csv"

    with pytest.raises(ValueError):
        contracts.write_coverage(output, rows, expected_count=expected_count)

    assert not output.exists()
    assert not (tmp_path / "q1_summary.json").exists()


def test_write_coverage_writes_success_and_failure_rows_with_summary(
    tmp_path: Path,
) -> None:
    contracts = _contracts()
    output = tmp_path / "q1_samples.csv"
    rows = [_coverage_row("one"), _coverage_row("two", status="failed")]

    counts = contracts.write_coverage(output, rows, expected_count=2)

    assert counts == {"coverage_count": 2, "success_count": 1, "failed_count": 1}
    with output.open(newline="") as coverage_file:
        reader = csv.DictReader(coverage_file)
        assert tuple(reader.fieldnames or ()) == COVERAGE_COLUMNS
        assert list(reader) == [
            {key: str(value) for key, value in rows[0].items()},
            {key: str(value) for key, value in rows[1].items()},
        ]
    assert json.loads((tmp_path / "q1_summary.json").read_text()) == counts


def test_write_coverage_rolls_back_when_summary_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts()
    output = tmp_path / "q1_samples.csv"
    summary = tmp_path / "q1_summary.json"
    original_replace = contracts.os.replace
    replace_calls = 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated summary publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(contracts.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="simulated summary publication failure"):
        contracts.write_coverage(output, [_coverage_row("one")], expected_count=1)

    assert replace_calls == 2
    assert not output.exists()
    assert not summary.exists()
    assert not list(tmp_path.glob(".q1_samples.csv.*.tmp"))
    assert not list(tmp_path.glob(".q1_summary.json.*.tmp"))


def test_write_coverage_preserves_external_csv_replacement_on_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts()
    output = tmp_path / "q1_samples.csv"
    summary = tmp_path / "q1_summary.json"
    external_csv = tmp_path / "external-q1_samples.csv"
    original_replace = contracts.os.replace
    replace_calls = 0

    def replace_then_fail_summary(source: object, destination: object) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            external_csv.write_text("external publisher output\n")
            original_replace(external_csv, output)
            raise OSError("simulated summary publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(contracts.os, "replace", replace_then_fail_summary)

    with pytest.raises(OSError, match="simulated summary publication failure"):
        contracts.write_coverage(output, [_coverage_row("one")], expected_count=1)

    assert replace_calls == 2
    assert output.read_text() == "external publisher output\n"
    assert not summary.exists()
    assert not list(tmp_path.glob(".q1_samples.csv.*.tmp"))
    assert not list(tmp_path.glob(".q1_summary.json.*.tmp"))


@pytest.mark.parametrize("existing_name", ["q1_samples.csv", "q1_summary.json"])
def test_write_coverage_refuses_to_overwrite_existing_outputs(
    tmp_path: Path, existing_name: str
) -> None:
    contracts = _contracts()
    output = tmp_path / "q1_samples.csv"
    existing = tmp_path / existing_name
    existing.write_text("preserve me")

    with pytest.raises(ValueError, match="already exists"):
        contracts.write_coverage(output, [_coverage_row("one")], expected_count=1)

    assert existing.read_text() == "preserve me"
    assert not (tmp_path / ({"q1_samples.csv", "q1_summary.json"} - {existing_name}).pop()).exists()
