from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from e_mosei_audit.q1 import extractors
from e_mosei_audit.q1.extractors import WordInterval, text_slot_features


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"word": "", "start": 0.0, "end": 0.1, "char_start": 0, "char_end": 0}, "word"),
        ({"word": "word", "start": float("nan"), "end": 0.1, "char_start": 0, "char_end": 4}, "time"),
        ({"word": "word", "start": 0.1, "end": 0.1, "char_start": 0, "char_end": 4}, "time"),
        ({"word": "word", "start": 0.0, "end": 0.1, "char_start": -1, "char_end": 4}, "character"),
        ({"word": "word", "start": 0.0, "end": 0.1, "char_start": 4, "char_end": 3}, "character"),
    ],
)
def test_word_interval_rejects_invalid_text_time_or_character_spans(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WordInterval(**kwargs)


def test_text_slot_features_pools_tokens_by_their_physical_word_intervals() -> None:
    embeddings = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    intervals = [
        WordInterval("hello", 0.0, 0.4, 0, 5),
        WordInterval("world", 0.6, 1.0, 6, 11),
    ]
    slots = np.array([[0.0, 0.5], [0.5, 1.0]])

    values, mask = text_slot_features(embeddings, intervals, slots)

    assert values.tolist() == [[1.0, 3.0], [5.0, 7.0]]
    assert mask.tolist() == [True, True]


def test_text_slot_features_uses_overlap_weighting_and_preserves_empty_slot_mask() -> None:
    embeddings = np.array([[2.0], [10.0]], dtype=np.float32)
    intervals = [
        WordInterval("first", 0.0, 0.75, 0, 5),
        WordInterval("second", 0.5, 1.0, 6, 12),
    ]
    slots = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]])

    values, mask = text_slot_features(embeddings, intervals, slots)

    np.testing.assert_allclose(values[:, 0], [2.0, 22.0 / 3.0, 0.0])
    assert mask.tolist() == [True, True, False]


def test_text_slot_features_rejects_embedding_interval_count_mismatch() -> None:
    embeddings = np.zeros((2, 3), dtype=np.float32)
    slots = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError, match="matching"):
        text_slot_features(embeddings, [WordInterval("one", 0.0, 1.0, 0, 3)], slots)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0], dtype=np.float32),
        np.array([[1]], dtype=np.int64),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[1.0 + 2.0j]], dtype=np.complex64),
    ],
)
def test_text_slot_features_rejects_malformed_embeddings(embeddings: np.ndarray) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        text_slot_features(
            embeddings,
            [WordInterval("one", 0.0, 1.0, 0, 3)],
            np.array([[0.0, 1.0]]),
        )


@pytest.mark.parametrize(
    ("builder_name", "arguments", "package"),
    [
        ("build_whisperx_aligner", (Path("models"),), "whisperx"),
        ("build_bert_encoder", (Path("models"),), "transformers"),
        ("build_opensmile_extractor", (), "opensmile"),
        ("build_mediapipe_extractor", (Path("models"),), "mediapipe"),
    ],
)
def test_builders_translate_missing_optional_packages_to_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    arguments: tuple[Path, ...],
    package: str,
) -> None:
    original_import_module = extractors.importlib.import_module

    def fail_target_package(name: str, package_: object = None) -> object:
        if name == package:
            raise ModuleNotFoundError(f"No module named {name}", name=name)
        return original_import_module(name, package_)

    monkeypatch.setattr(extractors.importlib, "import_module", fail_target_package)

    with pytest.raises(RuntimeError, match=package):
        getattr(extractors, builder_name)(*arguments)


def test_normalize_bert_embeddings_requires_token_level_768_dimensions() -> None:
    values = extractors.normalize_bert_embeddings(np.ones((2, 768), dtype=np.float64))

    assert values.shape == (2, 768)
    assert values.dtype == np.float32
    with pytest.raises(ValueError, match="768"):
        extractors.normalize_bert_embeddings(np.ones((2, 767), dtype=np.float32))


def test_normalize_mediapipe_frame_pads_blendshapes_and_emits_56_dimensions() -> None:
    values = extractors.normalize_mediapipe_frame(
        np.arange(3, dtype=np.float32),
        face_area=0.2,
        center_x=0.5,
        center_y=0.6,
        detector_score=0.9,
    )

    assert values.shape == (56,)
    assert values.dtype == np.float32
    assert values[:5].tolist() == [0.0, 1.0, 2.0, 0.0, 0.0]
    assert values[-4:].tolist() == pytest.approx([0.2, 0.5, 0.6, 0.9])


def test_normalize_opensmile_windows_requires_exactly_25_lld_dimensions() -> None:
    values, starts, ends = extractors.normalize_opensmile_windows(
        np.ones((2, 25), dtype=np.float64),
        np.array([0.0, 0.01]),
        np.array([0.025, 0.035]),
    )

    assert values.shape == (2, 25)
    assert values.dtype == np.float32
    assert starts.tolist() == [0.0, 0.01]
    assert ends.tolist() == [0.025, 0.035]
    with pytest.raises(ValueError, match="25"):
        extractors.normalize_opensmile_windows(
            np.ones((1, 24), dtype=np.float32),
            np.array([0.0]),
            np.array([0.025]),
        )
