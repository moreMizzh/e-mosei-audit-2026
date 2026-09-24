from __future__ import annotations

import numpy as np

from e_mosei_audit.q2.missingness import (
    ModalityMasks,
    ValidationScenario,
    apply_contiguous_drop,
    apply_validation_scenario,
    fit_normalizer,
    observed_masks,
    validation_scenarios,
)


def test_observed_masks_use_attention_and_zero_vector_evidence() -> None:
    text_bert = np.zeros((1, 3, 5), dtype=np.int64)
    text_bert[0, 0, :2] = [101, 102]
    text_bert[0, 1, :2] = 1
    audio = np.ones((1, 5, 74), dtype=np.float32)
    audio[:, 1] = 0
    audio[:, 2:] = 0
    vision = np.ones((1, 5, 35), dtype=np.float32)
    vision[:, 2:] = 0

    masks = observed_masks(text_bert, audio, vision)

    np.testing.assert_array_equal(masks.text, [[True, True, False, False, False]])
    np.testing.assert_array_equal(masks.audio, [[True, False, False, False, False]])
    np.testing.assert_array_equal(masks.vision, [[True, True, False, False, False]])
    np.testing.assert_array_equal(masks.temporal, [[True, True, False, False, False]])


def test_contiguous_drop_never_changes_padding_or_original_masks() -> None:
    original = ModalityMasks(
        text=np.array([[True, True, True, False, False]]),
        audio=np.array([[True, True, True, False, False]]),
        vision=np.array([[True, True, True, False, False]]),
        temporal=np.array([[True, True, True, False, False]]),
    )

    result = apply_contiguous_drop(
        original,
        rng=np.random.default_rng(7),
        modalities=("audio",),
        fraction_range=(0.5, 0.5),
    )

    np.testing.assert_array_equal(original.audio, [[True, True, True, False, False]])
    assert not result.masks.audio[0, 3:].any()
    assert result.drops[0].modality == "audio"
    assert result.drops[0].length == 2
    assert not result.masks.audio[0, result.drops[0].start : result.drops[0].start + 2].any()


def test_validation_scenarios_cover_every_modality_position_and_duration() -> None:
    scenarios = validation_scenarios()

    assert len(scenarios) == 27
    assert {(scenario.modality, scenario.position, scenario.fraction) for scenario in scenarios} == {
        (modality, position, fraction)
        for modality in ("text", "audio", "vision")
        for position in ("beginning", "middle", "end")
        for fraction in (0.1, 0.3, 0.5)
    }


def test_validation_middle_scenario_hides_center_of_valid_run() -> None:
    original = ModalityMasks(
        text=np.array([[True] * 10 + [False] * 2]),
        audio=np.array([[True] * 10 + [False] * 2]),
        vision=np.array([[True] * 10 + [False] * 2]),
        temporal=np.array([[True] * 10 + [False] * 2]),
    )

    result = apply_validation_scenario(
        original, ValidationScenario(modality="audio", position="middle", fraction=0.3)
    )

    np.testing.assert_array_equal(result.masks.audio[0], [True, True, True, False, False, False, True, True, True, True, False, False])
    assert result.drops == (result.drops[0],)
    assert (result.drops[0].start, result.drops[0].length) == (3, 3)
    np.testing.assert_array_equal(original.audio, [[True] * 10 + [False] * 2])


def test_fit_normalizer_uses_only_training_available_positions() -> None:
    masks = ModalityMasks(
        text=np.array([[True, True, False]]),
        audio=np.array([[True, True, False]]),
        vision=np.array([[True, True, False]]),
        temporal=np.array([[True, True, False]]),
    )
    audio = np.zeros((1, 3, 74), dtype=np.float32)
    vision = np.zeros((1, 3, 35), dtype=np.float32)
    audio[0, 0], audio[0, 1], audio[0, 2] = 1, 3, 999
    vision[0, 0], vision[0, 1], vision[0, 2] = 2, 6, 999

    normalizer = fit_normalizer(audio, vision, masks)

    np.testing.assert_allclose(normalizer.audio_mean, np.full(74, 2.0))
    np.testing.assert_allclose(normalizer.audio_std, np.full(74, 1.0))
    np.testing.assert_allclose(normalizer.vision_mean, np.full(35, 4.0))
    np.testing.assert_allclose(normalizer.vision_std, np.full(35, 2.0))
