from __future__ import annotations

import numpy as np


def make_slots(duration: float, count: int = 50) -> np.ndarray:
    if isinstance(duration, (bool, np.bool_)) or not isinstance(
        duration, (int, float, np.integer, np.floating)
    ):
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("count must be a positive integer")
    if count <= 0:
        raise ValueError("count must be a positive integer")

    boundaries = np.linspace(0.0, float(duration), count + 1)
    return np.column_stack((boundaries[:-1], boundaries[1:]))


def pool_intervals(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, starts, ends, slots = _validate_pool_inputs(values, starts, ends, slots)
    means, mask = _weighted_means(values, starts, ends, slots)
    return means.astype(np.float32), mask


def pool_moments(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, starts, ends, slots = _validate_pool_inputs(values, starts, ends, slots)
    means, mask = _weighted_means(values, starts, ends, slots)
    feature_count = values.shape[1]
    result = np.zeros((len(slots), feature_count * 2), dtype=np.float32)
    result[:, :feature_count] = means

    for index, (slot_start, slot_end) in enumerate(slots):
        overlap = np.maximum(
            0.0, np.minimum(ends, slot_end) - np.maximum(starts, slot_start)
        )
        total_overlap = overlap.sum()
        if total_overlap > 0:
            variance = (
                ((values - means[index]) ** 2 * overlap[:, None]).sum(axis=0)
                / total_overlap
            )
            result[index, feature_count:] = np.sqrt(variance)

    return result, mask


def _validate_pool_inputs(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values_array = np.asarray(values)
    starts_array = np.asarray(starts)
    ends_array = np.asarray(ends)
    slots_array = np.asarray(slots)

    if values_array.ndim != 2:
        raise ValueError("values must be a 2D feature matrix")
    if (
        starts_array.ndim != 1
        or ends_array.ndim != 1
        or starts_array.shape != ends_array.shape
        or starts_array.shape[0] != values_array.shape[0]
    ):
        raise ValueError("values and intervals must have matching leading dimensions")
    if slots_array.ndim != 2 or slots_array.shape[1] != 2:
        raise ValueError("slots must have shape (S, 2)")

    try:
        values_array = values_array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be numeric") from error
    starts_array = _as_finite_bounds(starts_array)
    ends_array = _as_finite_bounds(ends_array)
    slots_array = _as_finite_bounds(slots_array)

    if np.any(ends_array <= starts_array):
        raise ValueError("source intervals must have positive duration")
    if np.any(slots_array[:, 1] <= slots_array[:, 0]):
        raise ValueError("slot intervals must have positive duration")
    if np.any(slots_array[1:, 0] < slots_array[:-1, 1]):
        raise ValueError("slots must be monotonic and non-overlapping")
    if np.any(slots_array[1:, 0] > slots_array[:-1, 1]):
        raise ValueError("slots must be contiguous")

    return values_array, starts_array, ends_array, slots_array


def _as_finite_bounds(bounds: np.ndarray) -> np.ndarray:
    try:
        numeric_bounds = bounds.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("time bounds must be finite") from error

    if not np.all(np.isfinite(numeric_bounds)):
        raise ValueError("time bounds must be finite")
    return numeric_bounds


def _weighted_means(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(slots), values.shape[1]), dtype=np.float64)
    mask = np.zeros(len(slots), dtype=bool)

    for index, (slot_start, slot_end) in enumerate(slots):
        overlap = np.maximum(
            0.0, np.minimum(ends, slot_end) - np.maximum(starts, slot_start)
        )
        total_overlap = overlap.sum()
        if total_overlap > 0:
            means[index] = (values * overlap[:, None]).sum(axis=0) / total_overlap
            mask[index] = True

    return means, mask
