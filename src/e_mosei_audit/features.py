"""Feature-contract summaries for Attachment 2 pickle payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np


_SPLITS = ("train", "valid", "test")
_TEMPORAL_FIELDS = {"audio", "vision", "text", "text_bert"}


def audit_feature_dataset(
    payload: Mapping[str, Mapping[str, Any]], *, source_name: str, chunk_size: int = 32
) -> dict[str, object]:
    """Create a JSON-safe schema and quality summary for an Attachment 2 payload."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    result: dict[str, object] = {"source_name": source_name, "splits": {}, "errors": []}
    splits: dict[str, object] = {}
    errors: list[str] = []
    for split_name in _SPLITS:
        split = payload.get(split_name)
        if not isinstance(split, Mapping):
            errors.append(f"missing or invalid split: {split_name}")
            continue
        splits[split_name] = _audit_split(split, chunk_size)
    result["splits"] = splits
    result["errors"] = errors
    return result


def _audit_split(split: Mapping[str, Any], chunk_size: int) -> dict[str, object]:
    field_lengths = {
        name: length
        for name, value in split.items()
        if (length := _sample_length(value)) is not None
    }
    sample_count = _select_sample_count(split, field_lengths)
    inconsistent_fields = {
        name: length for name, length in field_lengths.items() if sample_count is not None and length != sample_count
    }
    fields = {
        name: _audit_field(name, value, split.get(f"{name}_lengths"), chunk_size)
        for name, value in sorted(split.items())
    }

    summary: dict[str, object] = {
        "sample_count": sample_count,
        "fields": fields,
        "inconsistent_fields": inconsistent_fields,
        "length_fields": {
            name: _length_statistics(value)
            for name, value in split.items()
            if "length" in name.lower() and _numeric_vector(value) is not None
        },
    }
    labels = split.get("classification_labels")
    if labels is not None:
        summary["label_distribution"] = _label_distribution(labels)
    return summary


def _sample_length(value: Any) -> int | None:
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _select_sample_count(split: Mapping[str, Any], field_lengths: Mapping[str, int]) -> int | None:
    identifier_length = _sample_length(split.get("id"))
    if identifier_length is not None:
        return identifier_length
    if not field_lengths:
        return None
    counts = Counter(field_lengths.values())
    return max(counts, key=lambda length: (counts[length], -length))


def _audit_field(name: str, value: Any, valid_lengths: Any, chunk_size: int) -> dict[str, object]:
    if isinstance(value, np.ndarray):
        result: dict[str, object] = {
            "kind": "ndarray",
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
        }
        if np.issubdtype(value.dtype, np.number):
            result["nonfinite"] = _nonfinite_counts(value, chunk_size)
        zero_runs = _zero_run_summary(name, value, valid_lengths, chunk_size)
        if zero_runs is not None:
            result["zero_runs"] = zero_runs
        return result
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "length": len(value)}
    return {"kind": type(value).__name__}


def _nonfinite_counts(values: np.ndarray, chunk_size: int) -> dict[str, int]:
    if not np.issubdtype(values.dtype, np.inexact):
        return {"nan": 0, "posinf": 0, "neginf": 0}
    nan_count = posinf_count = neginf_count = 0
    if values.ndim == 0:
        batches = (values.reshape(1),)
    else:
        batches = (values[start : start + chunk_size] for start in range(0, values.shape[0], chunk_size))
    for batch in batches:
        nan_count += int(np.isnan(batch).sum())
        posinf_count += int(np.isposinf(batch).sum())
        neginf_count += int(np.isneginf(batch).sum())
    return {"nan": nan_count, "posinf": posinf_count, "neginf": neginf_count}


def _zero_run_summary(
    name: str, values: np.ndarray, valid_lengths: Any, chunk_size: int
) -> dict[str, int] | None:
    if name not in _TEMPORAL_FIELDS or not np.issubdtype(values.dtype, np.number):
        return None
    if name == "text_bert" and values.ndim >= 3:
        time_axis = 2
    elif values.ndim >= 3:
        time_axis = 1
    else:
        return None

    lengths = _normalise_lengths(valid_lengths, values.shape[0], values.shape[time_axis])
    samples_with_zero = leading_positions = trailing_positions = interior_positions = 0
    outside_valid_positions = longest_run = 0
    for start in range(0, values.shape[0], chunk_size):
        batch = values[start : start + chunk_size]
        feature_axes = tuple(axis for axis in range(1, batch.ndim) if axis != time_axis)
        zero_mask = np.all(batch == 0, axis=feature_axes)
        for batch_index, positions in enumerate(zero_mask, start=start):
            valid_length = lengths[batch_index] if lengths is not None else len(positions)
            active_positions = positions[:valid_length]
            outside_valid_positions += int(positions[valid_length:].sum())
            if not np.any(active_positions):
                continue
            samples_with_zero += 1
            leading = _leading_true_count(active_positions)
            trailing = (
                _leading_true_count(active_positions[::-1]) if leading < len(active_positions) else 0
            )
            leading_positions += leading
            trailing_positions += trailing
            interior_positions += int(active_positions.sum()) - leading - trailing
            longest_run = max(longest_run, _longest_true_run(active_positions))
    return {
        "samples_with_zero": samples_with_zero,
        "leading_positions": leading_positions,
        "trailing_positions": trailing_positions,
        "interior_positions": interior_positions,
        "outside_valid_positions": outside_valid_positions,
        "longest_run": longest_run,
    }


def _normalise_lengths(value: Any, sample_count: int, maximum: int) -> np.ndarray | None:
    lengths = _numeric_vector(value)
    if lengths is None or len(lengths) != sample_count:
        return None
    return np.clip(lengths.astype(np.int64, copy=False), 0, maximum)


def _leading_true_count(positions: np.ndarray) -> int:
    false_positions = np.flatnonzero(~positions)
    return int(false_positions[0]) if len(false_positions) else len(positions)


def _longest_true_run(positions: np.ndarray) -> int:
    longest = current = 0
    for position in positions:
        if position:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _numeric_vector(value: Any) -> np.ndarray | None:
    if not isinstance(value, (np.ndarray, list, tuple)):
        return None
    vector = np.asarray(value)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
        return None
    return vector


def _length_statistics(values: Any) -> dict[str, int | float | None]:
    flattened = _numeric_vector(values)
    if flattened is None:
        raise ValueError("length statistics require a one-dimensional numeric value")
    finite = flattened[np.isfinite(flattened)]
    if not len(finite):
        return {"min": None, "max": None, "mean": None}
    return {
        "min": _scalar(finite.min()),
        "max": _scalar(finite.max()),
        "mean": float(finite.mean()),
    }


def _label_distribution(values: Any) -> dict[str, int]:
    flattened = np.asarray(values).reshape(-1)
    counts: Counter[str] = Counter()
    for value in flattened:
        counts[str(_scalar(value))] += 1
    return dict(sorted(counts.items()))


def _scalar(value: Any) -> int | float | str | bool:
    result = value.item() if isinstance(value, np.generic) else value
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result
