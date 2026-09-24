"""Availability masks and contiguous-loss controls for Problem 2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ModalityMasks:
    """Per-position observed evidence for the three selected aligned modalities."""

    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    temporal: np.ndarray


ModalityName = Literal["text", "audio", "vision"]


@dataclass(frozen=True)
class SyntheticDrop:
    """One generated local unavailable interval and its observed-coverage provenance."""

    sample_index: int
    modality: ModalityName
    start: int
    length: int
    target_available_positions: int
    selected_run_length: int
    fraction: float

    @property
    def actual_available_fraction(self) -> float:
        """Return the actual share of all target-modality evidence hidden."""

        return self.length / self.target_available_positions


@dataclass(frozen=True)
class DroppedMasks:
    """Copied masks with synthetic unavailable intervals and their provenance."""

    masks: ModalityMasks
    drops: tuple[SyntheticDrop, ...]


@dataclass(frozen=True)
class ValidationScenario:
    """A deterministic validation-only local-missingness intervention."""

    modality: ModalityName
    position: Literal["beginning", "middle", "end"]
    fraction: float


@dataclass(frozen=True)
class FeatureNormalizer:
    """Training-only feature statistics for the numeric aligned modalities."""

    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        """Return JSON-compatible provenance without serialising raw training arrays."""

        return {
            "audio_mean": self.audio_mean.tolist(),
            "audio_std": self.audio_std.tolist(),
            "vision_mean": self.vision_mean.tolist(),
            "vision_std": self.vision_std.tolist(),
        }


def observed_masks(text_bert: np.ndarray, audio: np.ndarray, vision: np.ndarray) -> ModalityMasks:
    """Infer observed aligned positions from attention and all-zero feature evidence."""

    if text_bert.ndim != 3 or text_bert.shape[1] != 3:
        raise ValueError("text_bert must have shape [N, 3, T]")
    sample_count, _, positions = text_bert.shape
    if audio.shape != (sample_count, positions, 74):
        raise ValueError("audio must have shape [N, T, 74] matching text_bert")
    if vision.shape != (sample_count, positions, 35):
        raise ValueError("vision must have shape [N, T, 35] matching text_bert")
    text = text_bert[:, 1, :].astype(bool, copy=False)
    audio_available = np.any(audio != 0, axis=2)
    vision_available = np.any(vision != 0, axis=2)
    return ModalityMasks(
        text=text,
        audio=audio_available,
        vision=vision_available,
        temporal=text | audio_available | vision_available,
    )


def apply_contiguous_drop(
    masks: ModalityMasks,
    *,
    rng: np.random.Generator,
    modalities: tuple[ModalityName, ...],
    fraction_range: tuple[float, float] = (0.1, 0.5),
) -> DroppedMasks:
    """Hide contiguous regions inside observed time runs without mutating source masks."""

    _validate_masks(masks)
    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("modalities must contain one or more distinct modality names")
    if any(modality not in ("text", "audio", "vision") for modality in modalities):
        raise ValueError("modalities must only contain text, audio, or vision")
    low, high = fraction_range
    if not 0 < low <= high <= 1:
        raise ValueError("fraction_range must satisfy 0 < low <= high <= 1")

    text = masks.text.copy()
    audio = masks.audio.copy()
    vision = masks.vision.copy()
    fields = {"text": text, "audio": audio, "vision": vision}
    drops: list[SyntheticDrop] = []
    for sample_index in range(masks.temporal.shape[0]):
        for modality in modalities:
            positions = getattr(masks, modality)[sample_index]
            runs = _true_runs(positions)
            if not runs:
                continue
            run_start, run_length = runs[int(rng.integers(len(runs)))]
            fraction = low if low == high else float(rng.uniform(low, high))
            length = min(run_length, max(1, math.ceil(run_length * fraction)))
            offset = int(rng.integers(run_length - length + 1))
            start = run_start + offset
            fields[modality][sample_index, start : start + length] = False
            drops.append(
                SyntheticDrop(
                    sample_index=sample_index,
                    modality=modality,
                    start=start,
                    length=length,
                    target_available_positions=int(positions.sum()),
                    selected_run_length=run_length,
                    fraction=fraction,
                )
            )
    return DroppedMasks(
        masks=ModalityMasks(text=text, audio=audio, vision=vision, temporal=text | audio | vision),
        drops=tuple(drops),
    )


def validation_scenarios() -> tuple[ValidationScenario, ...]:
    """Return the complete deterministic 3 x 3 x 3 validation matrix."""

    return tuple(
        ValidationScenario(modality=modality, position=position, fraction=fraction)
        for modality in ("text", "audio", "vision")
        for position in ("beginning", "middle", "end")
        for fraction in (0.1, 0.3, 0.5)
    )


def apply_validation_scenario(masks: ModalityMasks, scenario: ValidationScenario) -> DroppedMasks:
    """Apply one fixed local-loss scenario to each sample without randomness."""

    _validate_masks(masks)
    text = masks.text.copy()
    audio = masks.audio.copy()
    vision = masks.vision.copy()
    fields = {"text": text, "audio": audio, "vision": vision}
    drops: list[SyntheticDrop] = []
    for sample_index in range(masks.temporal.shape[0]):
        positions = getattr(masks, scenario.modality)[sample_index]
        runs = _true_runs(positions)
        if not runs:
            continue
        run_start, run_length = max(runs, key=lambda run: (run[1], -run[0]))
        length = min(run_length, max(1, math.ceil(run_length * scenario.fraction)))
        if scenario.position == "beginning":
            start = run_start
        elif scenario.position == "middle":
            start = run_start + (run_length - length) // 2
        else:
            start = run_start + run_length - length
        fields[scenario.modality][sample_index, start : start + length] = False
        drops.append(
            SyntheticDrop(
                sample_index=sample_index,
                modality=scenario.modality,
                start=start,
                length=length,
                target_available_positions=int(positions.sum()),
                selected_run_length=run_length,
                fraction=scenario.fraction,
            )
        )
    return DroppedMasks(
        masks=ModalityMasks(text=text, audio=audio, vision=vision, temporal=text | audio | vision),
        drops=tuple(drops),
    )


def fit_normalizer(audio: np.ndarray, vision: np.ndarray, masks: ModalityMasks) -> FeatureNormalizer:
    """Fit numeric modality statistics on observed training positions only."""

    _validate_masks(masks)
    sample_count, positions = masks.temporal.shape
    if audio.shape != (sample_count, positions, 74):
        raise ValueError("audio must have shape [N, T, 74] matching masks")
    if vision.shape != (sample_count, positions, 35):
        raise ValueError("vision must have shape [N, T, 35] matching masks")
    audio_values = audio[masks.temporal & masks.audio]
    vision_values = vision[masks.temporal & masks.vision]
    if not len(audio_values) or not len(vision_values):
        raise ValueError("training masks must retain at least one audio and vision position")
    audio_mean, audio_std = _mean_and_std(audio_values)
    vision_mean, vision_std = _mean_and_std(vision_values)
    return FeatureNormalizer(
        audio_mean=audio_mean,
        audio_std=audio_std,
        vision_mean=vision_mean,
        vision_std=vision_std,
    )


def _validate_masks(masks: ModalityMasks) -> None:
    shape = masks.temporal.shape
    if len(shape) != 2 or any(field.shape != shape for field in (masks.text, masks.audio, masks.vision)):
        raise ValueError("all modality masks must share shape [N, T]")
    if any(field.dtype != np.bool_ for field in (masks.text, masks.audio, masks.vision, masks.temporal)):
        raise ValueError("all modality masks must have bool dtype")


def _true_runs(positions: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, position in enumerate(positions):
        if position and start is None:
            start = index
        elif not position and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(positions) - start))
    return runs


def _mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    safe_std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), safe_std.astype(np.float32)
