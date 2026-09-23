from __future__ import annotations

import numpy as np

from e_mosei_audit.special import audit_special_payload


def test_audits_nested_attachment_three_payload_by_batch_sample() -> None:
    payload = {
        "test": {
            "audio": np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
            "text_bert": np.array([[[101, 102, 0], [1, 1, 0], [0, 0, 0]]]),
        }
    }

    result = audit_special_payload(
        payload,
        attachment="attachment3",
        version="aligned",
        source_file="附件3_01.pkl",
    )

    assert result.errors == []
    assert result.records[0]["sample_id"] == "attachment3:aligned:附件3_01"
    assert result.records[0]["layout"] == "nested_test"
    assert result.records[0]["fields"]["audio"]["zero_runs"] == {
        "leading_positions": 1,
        "trailing_positions": 1,
        "interior_positions": 0,
        "longest_run": 1,
    }


def test_audits_attachment_four_single_sample_with_its_provided_id() -> None:
    payload = {
        "id": "01",
        "raw_text": "A complete sentence.",
        "audio": np.array([[1.0, 1.0], [0.0, 0.0]]),
    }

    result = audit_special_payload(
        payload,
        attachment="attachment4",
        version="aligned",
        source_file="01.pkl",
    )

    assert result.errors == []
    assert result.records[0]["sample_id"] == "01"
    assert result.records[0]["layout"] == "single_sample"
    assert result.records[0]["fields"]["audio"]["shape"] == [2, 2]


def test_rejects_nested_fields_with_inconsistent_batch_dimensions() -> None:
    payload = {
        "test": {
            "id": np.array(["a", "b"]),
            "audio": np.ones((2, 3, 2)),
            "vision": np.ones((1, 3, 2)),
        }
    }

    result = audit_special_payload(
        payload,
        attachment="attachment3",
        version="aligned",
        source_file="附件3_bad.pkl",
    )

    assert result.records == []
    assert result.errors == ["附件3_bad.pkl: inconsistent nested sample count for vision: 1 != 2"]
