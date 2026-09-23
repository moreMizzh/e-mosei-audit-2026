from __future__ import annotations

import numpy as np
import pytest

from e_mosei_audit.q1.timeline import make_slots, pool_intervals, pool_moments


def test_make_slots_divides_duration_into_contiguous_physical_intervals() -> None:
    slots = make_slots(10.0)

    assert slots.shape == (50, 2)
    assert np.issubdtype(slots.dtype, np.floating)
    assert slots[0].tolist() == [0.0, 0.2]
    assert slots[-1].tolist() == [9.8, 10.0]
    np.testing.assert_array_equal(slots[1:, 0], slots[:-1, 1])


@pytest.mark.parametrize(
    ("duration", "count", "message"),
    [
        (0.0, 50, "duration"),
        (float("nan"), 50, "duration"),
        (10.0, 0, "count"),
        (10.0, 1.5, "count"),
        (10.0, True, "count"),
    ],
)
def test_make_slots_rejects_invalid_duration_or_count(
    duration: float, count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_slots(duration, count)


def test_pool_intervals_uses_overlap_duration_not_interval_midpoints() -> None:
    values = np.array([[2.0], [10.0]], dtype=np.float32)
    starts = np.array([0.0, 0.5])
    ends = np.array([0.5, 1.0])
    slots = np.array([[0.25, 0.75], [0.75, 1.0]])

    pooled, mask = pool_intervals(values, starts, ends, slots)

    assert pooled.dtype == np.float32
    np.testing.assert_allclose(pooled[:, 0], [6.0, 10.0])
    assert mask.tolist() == [True, True]


def test_pool_intervals_distinguishes_zero_features_from_empty_slots() -> None:
    values = np.zeros((1, 2), dtype=np.float32)

    pooled, mask = pool_intervals(
        values,
        np.array([0.0]),
        np.array([0.2]),
        np.array([[0.0, 0.2], [0.2, 0.4]]),
    )

    assert pooled.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert mask.tolist() == [True, False]


def test_pool_moments_concatenates_weighted_mean_and_population_deviation() -> None:
    values = np.array([[2.0], [6.0]], dtype=np.float32)

    pooled, mask = pool_moments(
        values,
        np.array([0.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([[0.0, 1.0]]),
    )

    assert pooled.dtype == np.float32
    assert pooled.tolist() == [[4.0, 2.0]]
    assert mask.tolist() == [True]


def test_pool_moments_returns_zeros_for_an_empty_slot() -> None:
    pooled, mask = pool_moments(
        np.array([[2.0]], dtype=np.float32),
        np.array([0.0]),
        np.array([0.2]),
        np.array([[0.0, 0.2], [0.2, 0.4]]),
    )

    assert pooled.tolist() == [[2.0, 0.0], [0.0, 0.0]]
    assert mask.tolist() == [True, False]


@pytest.mark.parametrize("pool", [pool_intervals, pool_moments])
@pytest.mark.parametrize(
    ("values", "starts", "ends", "slots", "message"),
    [
        (
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.0, 1.0]]),
            "values",
        ),
        (
            np.zeros((2, 1)),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.0, 1.0]]),
            "leading dimensions",
        ),
        (
            np.zeros((1, 1)),
            np.array([np.nan]),
            np.array([1.0]),
            np.array([[0.0, 1.0]]),
            "bounds",
        ),
        (
            np.zeros((1, 1)),
            np.array([0.0]),
            np.array([0.0]),
            np.array([[0.0, 1.0]]),
            "source intervals",
        ),
        (
            np.zeros((1, 1)),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.0, 0.0]]),
            "slot intervals",
        ),
        (
            np.zeros((1, 1)),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[0.0, 0.75], [0.5, 1.0]]),
            "non-overlapping",
        ),
    ],
)
def test_pooling_rejects_invalid_arrays_and_time_bounds(
    pool: object,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    slots: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pool(values, starts, ends, slots)
