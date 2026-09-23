"""Contract audits for Attachments 3 and 4 specialty test samples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_TEMPORAL_FIELDS = {"audio", "vision", "text", "text_bert"}


@dataclass
class SpecialAuditResult:
    records: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def audit_special_payload(
    payload: Mapping[str, Any], *, attachment: str, version: str, source_file: str
) -> SpecialAuditResult:
    """Summarize one specialty pickle without assigning a sentiment prediction."""

    result = SpecialAuditResult()
    if not isinstance(payload, Mapping):
        result.errors.append(f"{source_file}: payload is not a dictionary")
        return result

    nested = payload.get("test")
    if isinstance(nested, Mapping):
        layout = "nested_test"
        fields: Mapping[str, Any] = nested
        sample_count = _batch_size(fields)
    else:
        layout = "single_sample"
        fields = payload
        sample_count = 1

    if sample_count is None or sample_count < 1:
        result.errors.append(f"{source_file}: could not determine a positive sample count")
        return result

    for index in range(sample_count):
        values = {name: _value_for_sample(value, index, sample_count, layout) for name, value in fields.items()}
        sample_id = _sample_id(values.get("id"), attachment, version, source_file, index, sample_count)
        result.records.append(
            {
                "attachment": attachment,
                "version": version,
                "source_file": source_file,
                "sample_id": sample_id,
                "layout": layout,
                "fields": {name: _summarize_value(name, value) for name, value in sorted(values.items())},
            }
        )
    return result


def _batch_size(fields: Mapping[str, Any]) -> int | None:
    lengths = [
        int(value.shape[0])
        for value in fields.values()
        if isinstance(value, np.ndarray) and value.ndim >= 1
    ]
    if not lengths:
        return 1
    counts = Counter(lengths)
    return max(counts, key=lambda length: (counts[length], -length))


def _value_for_sample(value: Any, index: int, sample_count: int, layout: str) -> Any:
    if layout == "nested_test" and isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == sample_count:
        return value[index]
    if layout == "nested_test" and isinstance(value, (list, tuple)) and len(value) == sample_count:
        return value[index]
    return value


def _sample_id(
    provided_id: Any, attachment: str, version: str, source_file: str, index: int, sample_count: int
) -> str:
    if provided_id is not None:
        if isinstance(provided_id, np.generic):
            provided_id = provided_id.item()
        return str(provided_id)
    fallback = f"{attachment}:{version}:{Path(source_file).stem}"
    return fallback if sample_count == 1 else f"{fallback}:{index:03d}"


def _summarize_value(name: str, value: Any) -> dict[str, object]:
    if not isinstance(value, np.ndarray):
        return {"kind": type(value).__name__}
    summary: dict[str, object] = {
        "kind": "ndarray",
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
    }
    zero_runs = _zero_runs(name, value)
    if zero_runs is not None:
        summary["zero_runs"] = zero_runs
    return summary


def _zero_runs(name: str, values: np.ndarray) -> dict[str, int] | None:
    if name not in _TEMPORAL_FIELDS or not np.issubdtype(values.dtype, np.number):
        return None
    if name == "text_bert" and values.ndim >= 2:
        positions = np.all(values == 0, axis=0)
    elif values.ndim >= 2:
        positions = np.all(values == 0, axis=tuple(range(1, values.ndim)))
    else:
        return None
    leading = _leading_true_count(positions)
    trailing = _leading_true_count(positions[::-1]) if leading < len(positions) else 0
    return {
        "leading_positions": leading,
        "trailing_positions": trailing,
        "interior_positions": int(positions.sum()) - leading - trailing,
        "longest_run": _longest_true_run(positions),
    }


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
