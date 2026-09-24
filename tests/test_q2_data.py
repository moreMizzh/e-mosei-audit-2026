from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import pickle

import numpy as np
import pytest

from e_mosei_audit.archive import ArchiveMember
from e_mosei_audit.q2.data import (
    ALIGNED_50_MEMBER,
    DataContractError,
    aligned_split_from_mapping,
    load_aligned_dataset,
    load_attachment3_aligned,
    normalise_text_bert,
)


class FakeArchive:
    def __init__(self, members: dict[str, object]) -> None:
        self._members = {path: pickle.dumps(payload) for path, payload in members.items()}
        self.opened: list[str] = []

    def list_members(self) -> list[ArchiveMember]:
        return [
            ArchiveMember(path=path, size=len(payload), packed_size=len(payload), is_directory=False)
            for path, payload in self._members.items()
        ]

    @contextmanager
    def open_member(self, member_path: str):
        self.opened.append(member_path)
        yield BytesIO(self._members[member_path])


def attachment3_payload(token: int) -> dict[str, dict[str, np.ndarray]]:
    text_bert = np.zeros((1, 3, 50), dtype=np.float32)
    text_bert[0, 0, :2] = [101, token]
    text_bert[0, 1, :2] = 1
    return {
        "test": {
            "text_bert": text_bert,
            "audio": np.ones((1, 50, 74), dtype=np.float32),
            "vision": np.ones((1, 50, 35), dtype=np.float32),
        }
    }


def aligned_split_mapping(sample_count: int = 2) -> dict[str, object]:
    text_bert = np.zeros((sample_count, 3, 50), dtype=np.int64)
    text_bert[:, 0, :2] = [101, 102]
    text_bert[:, 1, :2] = 1
    return {
        "text_bert": text_bert,
        "audio": np.ones((sample_count, 50, 74), dtype=np.float64),
        "vision": np.ones((sample_count, 50, 35), dtype=np.float64),
        "classification_labels": np.array([0.0, 2.0][:sample_count]),
        "regression_labels": np.array([-1.0, 1.0][:sample_count]),
        "id": [f"sample-{index}" for index in range(sample_count)],
    }


def test_normalise_text_bert_accepts_integral_float_attachment3_tokens() -> None:
    raw = np.zeros((1, 3, 50), dtype=np.float32)
    raw[0, 0, :2] = [101.0, 102.0]
    raw[0, 1, :2] = 1.0

    tokens = normalise_text_bert(raw)

    assert tokens.dtype == np.int64
    np.testing.assert_array_equal(tokens, raw.astype(np.int64))


def test_normalise_text_bert_rejects_non_integral_tokens() -> None:
    raw = np.zeros((1, 3, 50), dtype=np.float32)
    raw[0, 0, 1] = 101.5

    with pytest.raises(DataContractError, match="integer-valued"):
        normalise_text_bert(raw)


def test_load_attachment3_aligned_naturally_sorts_members_and_normalises_tokens() -> None:
    archive = FakeArchive(
        {
            "E题数据/附件3-模态缺失特征样本/对齐版本/附件3_10.pkl": attachment3_payload(110),
            "E题数据/附件3-模态缺失特征样本/对齐版本/附件3_02.pkl": attachment3_payload(102),
        }
    )

    samples = load_attachment3_aligned(archive)

    assert [sample.sample_id for sample in samples] == [
        "attachment3:aligned:附件3_02",
        "attachment3:aligned:附件3_10",
    ]
    assert samples[0].text_bert.dtype == np.int64
    assert samples[0].text_bert[0, 0, 1] == 102


def test_aligned_split_from_mapping_preserves_sample_alignment() -> None:
    split = aligned_split_from_mapping(aligned_split_mapping(), split_name="train")

    assert split.sample_count == 2
    assert split.audio.shape == (2, 50, 74)
    assert split.vision.shape == (2, 50, 35)
    assert split.text_bert.dtype == np.int64
    np.testing.assert_array_equal(split.classification_labels, [0, 2])
    assert split.ids == ("sample-0", "sample-1")


def test_aligned_split_rejects_class_outside_three_polarities() -> None:
    fields = aligned_split_mapping()
    fields["classification_labels"] = np.array([0.0, 3.0])

    with pytest.raises(DataContractError, match="classification_labels"):
        aligned_split_from_mapping(fields, split_name="train")


def test_load_aligned_dataset_reads_only_selected_feature_member() -> None:
    archive = FakeArchive(
        {
            ALIGNED_50_MEMBER: {
                "train": aligned_split_mapping(2),
                "valid": aligned_split_mapping(1),
                "test": aligned_split_mapping(1),
            }
        }
    )

    dataset = load_aligned_dataset(archive)

    assert archive.opened == [ALIGNED_50_MEMBER]
    assert dataset.train.ids == ("sample-0", "sample-1")
    assert dataset.valid.sample_count == 1
    assert dataset.test.sample_count == 1


def test_aligned_split_rejects_nonfinite_audio_feature() -> None:
    fields = aligned_split_mapping()
    fields["audio"][0, 0, 0] = np.nan

    with pytest.raises(DataContractError, match="audio"):
        aligned_split_from_mapping(fields, split_name="train")
