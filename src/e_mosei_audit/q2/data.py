"""Read-only data contracts shared by Problem 2 training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any, Protocol

import numpy as np


ALIGNED_50_MEMBER = "E题数据/附件2-数据集特征文件/aligned_50.pkl"


class DataContractError(ValueError):
    """Raised when a competition feature payload violates the selected interface."""


class ArchiveReader(Protocol):
    """Read-only subset of the archive interface used by Q2 loaders."""

    def list_members(self) -> list[Any]: ...

    def open_member(self, member_path: str) -> Any: ...


@dataclass(frozen=True)
class Attachment3Sample:
    """One aligned, unlabelled specialty sample and its stable archive provenance."""

    sample_id: str
    source_file: str
    text_bert: np.ndarray
    audio: np.ndarray
    vision: np.ndarray


@dataclass(frozen=True)
class AlignedSplit:
    """One Attachment 2 aligned split with all arrays indexed by sample."""

    split_name: str
    ids: tuple[str, ...]
    text_bert: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    classification_labels: np.ndarray
    regression_labels: np.ndarray

    @property
    def sample_count(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class AlignedDataset:
    """The three standard Attachment 2 aligned splits."""

    train: AlignedSplit
    valid: AlignedSplit
    test: AlignedSplit


def normalise_text_bert(value: np.ndarray) -> np.ndarray:
    """Validate BERT's three-row token interface and return int64 tokens."""

    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[1:] != (3, 50):
        raise DataContractError("text_bert must have shape [N, 3, 50]")
    if not np.issubdtype(value.dtype, np.number):
        raise DataContractError("text_bert must be numeric")
    if not np.isfinite(value).all() or not np.equal(value, np.rint(value)).all():
        raise DataContractError("text_bert must contain finite integer-valued tokens")
    return value.astype(np.int64, copy=False)


def aligned_split_from_mapping(fields: Mapping[str, Any], *, split_name: str) -> AlignedSplit:
    """Validate one aligned Attachment 2 split and discard unused legacy fields."""

    identifiers = fields.get("id")
    if not isinstance(identifiers, (list, tuple)) or not identifiers:
        raise DataContractError(f"{split_name}: id must be a non-empty sequence")
    ids = tuple(str(value) for value in identifiers)
    sample_count = len(ids)
    text_bert = normalise_text_bert(_require_text_bert_array(fields, sample_count, split_name))
    audio = _normalise_batch_feature(
        _require_batch_array(fields, "audio", sample_count, 50, 74, split_name), "audio", split_name
    )
    vision = _normalise_batch_feature(
        _require_batch_array(fields, "vision", sample_count, 50, 35, split_name), "vision", split_name
    )
    classification_labels = _normalise_classification_labels(
        fields.get("classification_labels"), sample_count, split_name
    )
    regression_labels = _normalise_vector(
        fields.get("regression_labels"), "regression_labels", sample_count, split_name, np.float32
    )
    return AlignedSplit(
        split_name=split_name,
        ids=ids,
        text_bert=text_bert,
        audio=audio,
        vision=vision,
        classification_labels=classification_labels,
        regression_labels=regression_labels,
    )


def load_aligned_dataset(archive: ArchiveReader) -> AlignedDataset:
    """Load the sole selected Attachment 2 feature version from the archive stream."""

    with archive.open_member(ALIGNED_50_MEMBER) as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise DataContractError("aligned_50.pkl: expected a mapping of standard splits")
    splits: dict[str, AlignedSplit] = {}
    for split_name in ("train", "valid", "test"):
        fields = payload.get(split_name)
        if not isinstance(fields, Mapping):
            raise DataContractError(f"aligned_50.pkl: missing mapping split {split_name}")
        splits[split_name] = aligned_split_from_mapping(fields, split_name=split_name)
    return AlignedDataset(train=splits["train"], valid=splits["valid"], test=splits["test"])


def load_attachment3_aligned(archive: ArchiveReader) -> list[Attachment3Sample]:
    """Read aligned Attachment 3 members in filename-number order without extraction."""

    members = [
        member
        for member in archive.list_members()
        if not getattr(member, "is_directory", False)
        and "附件3-模态缺失特征样本/对齐版本/" in member.path
        and member.path.endswith(".pkl")
    ]
    members.sort(key=lambda member: _attachment3_member_key(member.path))
    if not members:
        raise DataContractError("no aligned Attachment 3 pickle members were found")

    samples: list[Attachment3Sample] = []
    for member in members:
        with archive.open_member(member.path) as stream:
            payload = pickle.load(stream)
        samples.append(_attachment3_sample_from_payload(payload, member.path))
    return samples


def _attachment3_member_key(member_path: str) -> int:
    match = re.search(r"附件3_(\d+)\.pkl$", member_path)
    if match is None:
        raise DataContractError(f"unexpected aligned Attachment 3 member name: {member_path}")
    return int(match.group(1))


def _attachment3_sample_from_payload(payload: Any, member_path: str) -> Attachment3Sample:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("test"), Mapping):
        raise DataContractError(f"{member_path}: expected a mapping with nested test fields")
    fields = payload["test"]
    source_file = Path(member_path).name
    stem = Path(member_path).stem
    return Attachment3Sample(
        sample_id=f"attachment3:aligned:{stem}",
        source_file=source_file,
        text_bert=normalise_text_bert(_require_array(fields, "text_bert", (1, 3, 50), member_path)),
        audio=_normalise_feature(_require_array(fields, "audio", (1, 50, 74), member_path), "audio", member_path),
        vision=_normalise_feature(
            _require_array(fields, "vision", (1, 50, 35), member_path), "vision", member_path
        ),
    )


def _require_array(fields: Mapping[str, Any], name: str, shape: tuple[int, ...], member_path: str) -> np.ndarray:
    value = fields.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise DataContractError(f"{member_path}: {name} must have shape {list(shape)}")
    return value


def _require_batch_array(
    fields: Mapping[str, Any], name: str, sample_count: int, positions: int, width: int, split_name: str
) -> np.ndarray:
    value = fields.get(name)
    expected_shape = (sample_count, positions, width)
    if not isinstance(value, np.ndarray) or value.shape != expected_shape:
        raise DataContractError(f"{split_name}: {name} must have shape {list(expected_shape)}")
    return value


def _require_text_bert_array(fields: Mapping[str, Any], sample_count: int, split_name: str) -> np.ndarray:
    value = fields.get("text_bert")
    expected_shape = (sample_count, 3, 50)
    if not isinstance(value, np.ndarray) or value.shape != expected_shape:
        raise DataContractError(f"{split_name}: text_bert must have shape {list(expected_shape)}")
    return value


def _normalise_feature(value: np.ndarray, name: str, member_path: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise DataContractError(f"{member_path}: {name} must contain finite numeric values")
    return value.astype(np.float32, copy=False)


def _normalise_batch_feature(value: np.ndarray, name: str, split_name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise DataContractError(f"{split_name}: {name} must contain finite numeric values")
    return value.astype(np.float32, copy=False)


def _normalise_vector(
    value: Any, name: str, sample_count: int, split_name: str, dtype: np.dtype[Any]
) -> np.ndarray:
    vector = np.asarray(value)
    if vector.shape != (sample_count,) or not np.issubdtype(vector.dtype, np.number):
        raise DataContractError(f"{split_name}: {name} must be a numeric vector of length {sample_count}")
    return vector.astype(dtype, copy=False)


def _normalise_classification_labels(value: Any, sample_count: int, split_name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.shape != (sample_count,) or not np.issubdtype(vector.dtype, np.number):
        raise DataContractError(f"{split_name}: classification_labels must be a numeric vector of length {sample_count}")
    if not np.isfinite(vector).all() or not np.equal(vector, np.rint(vector)).all():
        raise DataContractError(f"{split_name}: classification_labels must contain finite integers")
    labels = vector.astype(np.int64, copy=False)
    if not np.isin(labels, (0, 1, 2)).all():
        raise DataContractError(f"{split_name}: classification_labels must be in [0, 1, 2]")
    return labels
