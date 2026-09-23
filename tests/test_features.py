from __future__ import annotations

import numpy as np

from e_mosei_audit.features import audit_feature_dataset


def test_summarizes_splits_labels_nonfinite_values_and_zero_runs() -> None:
    payload = {
        "train": {
            "id": np.array(["a", "b"]),
            "audio": np.array(
                [
                    [[0.0, 0.0], [1.0, np.nan], [0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "classification_labels": np.array([0, 1]),
            "regression_labels": np.array([0.5, -1.0]),
            "audio_lengths": np.array([3, 3]),
        },
        "valid": {"id": np.array(["c"]), "classification_labels": np.array([1])},
        "test": {"id": np.array(["d"]), "classification_labels": np.array([0])},
    }

    result = audit_feature_dataset(payload, source_name="aligned_50.pkl", chunk_size=1)

    train = result["splits"]["train"]
    assert train["sample_count"] == 2
    assert train["label_distribution"] == {"0": 1, "1": 1}
    assert train["length_fields"]["audio_lengths"] == {"min": 3, "max": 3, "mean": 3.0}
    assert train["fields"]["audio"]["nonfinite"] == {"nan": 1, "posinf": 0, "neginf": 0}
    assert train["fields"]["audio"]["zero_runs"] == {
        "samples_with_zero": 2,
        "leading_positions": 1,
        "trailing_positions": 1,
        "interior_positions": 1,
        "outside_valid_positions": 2,
        "longest_run": 1,
    }


def test_records_field_lengths_that_do_not_match_the_split_sample_count() -> None:
    payload = {
        "train": {"id": np.array(["a", "b"]), "audio": np.zeros((3, 2, 2))},
        "valid": {"id": np.array(["c"])},
        "test": {"id": np.array(["d"])},
    }

    result = audit_feature_dataset(payload, source_name="fixture.pkl")

    assert result["splits"]["train"]["sample_count"] == 2
    assert result["splits"]["train"]["inconsistent_fields"] == {"audio": 3}
