from __future__ import annotations

import csv
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from e_mosei_audit.archive import ArchiveMember
from e_mosei_audit.q2.config import Q2Config
from e_mosei_audit.q2.data import ALIGNED_50_MEMBER
from e_mosei_audit.q2.runner import (
    Prediction,
    _publish_staging,
    _validate_output_target,
    check_q2,
    compute_metrics,
    run_q2,
    write_predictions,
)


def test_metric_summary_reports_accuracy_macro_f1_mae_and_pearson() -> None:
    metrics = compute_metrics(
        true_classes=np.array([0, 1, 2]),
        predicted_classes=np.array([0, 1, 2]),
        true_scores=np.array([-1.0, 0.0, 1.0]),
        predicted_scores=np.array([-1.0, 0.0, 1.0]),
    )

    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "mae": 0.0, "pearson": 1.0}


def test_metric_summary_returns_none_pearson_for_constant_targets() -> None:
    metrics = compute_metrics(
        true_classes=np.array([1, 1]),
        predicted_classes=np.array([1, 1]),
        true_scores=np.array([0.0, 0.0]),
        predicted_scores=np.array([0.2, 0.1]),
    )

    assert metrics["pearson"] is None


def test_prediction_csv_uses_fixed_polarity_mapping(tmp_path: Path) -> None:
    write_predictions(
        tmp_path,
        [Prediction(sample_id="x", source_file="附件3_01.pkl", polarity_class=0, sentiment_intensity=-0.5)],
    )

    with (tmp_path / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows == [
        {
            "sample_id": "x",
            "source_file": "附件3_01.pkl",
            "polarity_class": "0",
            "polarity_label": "Negative",
            "sentiment_intensity": "-0.5",
        }
    ]


class RunnerArchive:
    def __init__(self, members: dict[str, object]) -> None:
        self._contents = {path: pickle.dumps(payload) for path, payload in members.items()}
        self.verify_count = 0

    def verify(self) -> None:
        self.verify_count += 1

    def list_members(self) -> list[ArchiveMember]:
        return [
            ArchiveMember(path=path, size=len(content), packed_size=len(content), is_directory=False)
            for path, content in self._contents.items()
        ]

    @contextmanager
    def open_member(self, member_path: str):
        yield BytesIO(self._contents[member_path])


class TinyTokenEncoder:
    def encode(self, token_rows: torch.Tensor) -> torch.Tensor:
        embeddings = torch.zeros(token_rows.shape[0], 50, 768, device=token_rows.device)
        embeddings[:, :, 0] = token_rows[:, 0, :].to(torch.float32)
        return embeddings


def runner_split(labels: list[int]) -> dict[str, object]:
    sample_count = len(labels)
    tokens = np.zeros((sample_count, 3, 50), dtype=np.int64)
    tokens[:, 0, :4] = [101, 1000, 1001, 102]
    tokens[:, 1, :4] = 1
    values = np.arange(sample_count, dtype=np.float32).reshape(sample_count, 1, 1) + 1
    return {
        "id": [f"sample-{index}" for index in range(sample_count)],
        "text_bert": tokens,
        "audio": np.broadcast_to(values, (sample_count, 50, 74)).astype(np.float32).copy(),
        "vision": np.broadcast_to(values, (sample_count, 50, 35)).astype(np.float32).copy(),
        "classification_labels": np.asarray(labels, dtype=np.int64),
        "regression_labels": np.asarray([-1.0, 0.0, 1.0][:sample_count], dtype=np.float32),
    }


def runner_attachment3_payload(index: int) -> dict[str, dict[str, np.ndarray]]:
    tokens = np.zeros((1, 3, 50), dtype=np.float32)
    tokens[0, 0, :4] = [101, 2000 + index, 2001 + index, 102]
    tokens[0, 1, :4] = 1
    return {
        "test": {
            "text_bert": tokens,
            "audio": np.ones((1, 50, 74), dtype=np.float32),
            "vision": np.ones((1, 50, 35), dtype=np.float32),
        }
    }


def runner_config(tmp_path: Path) -> Q2Config:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"archive")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    bert_model = tmp_path / "bert"
    bert_model.mkdir()
    return Q2Config(
        archive=archive,
        seven_zip=seven_zip,
        bert_model=bert_model,
        output_dir=tmp_path / "q2-output",
        seed=7,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        weight_decay=0.0,
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        device="cpu",
    )


def test_run_q2_writes_30_attachment_predictions_and_27_scenarios(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": runner_split([0, 1, 2]),
        }
    }
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with (config.output_dir / "validation_scenarios.csv").open(encoding="utf-8", newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    metrics = json.loads((config.output_dir / "metrics.json").read_text(encoding="utf-8"))
    report = (config.output_dir / "audit_report.md").read_text(encoding="utf-8")
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert len(scenarios) == 27
    assert set(metrics["clean"]) == {"accuracy", "macro_f1", "mae", "pearson"}
    assert {
        "target_available_positions",
        "dropped_available_positions",
        "actual_available_fraction",
    } <= set(scenarios[0])
    assert all(float(row["actual_available_fraction"]) > 0 for row in scenarios)
    assert (config.output_dir / "model.pt").is_file()
    assert archive.verify_count == 1
    assert "缺失影响汇总" in report
    assert "最不利 macro-F1 场景" in report


def test_run_q2_does_not_validate_or_use_attachment2_test_split(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a training or validation input"},
        }
    }
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)

    summary = run_q2(runner_config(tmp_path), archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    assert summary["attachment3_count"] == 30


def test_check_q2_verifies_inputs_without_training_or_creating_output(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a Q2 runtime input"},
        }
    }
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)

    summary = check_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    assert summary == {"train_count": 3, "valid_count": 3, "attachment3_count": 30}
    assert archive.verify_count == 1
    assert not config.output_dir.exists()


def test_output_validation_requires_existing_parent_without_creating_it(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing-parent" / "q2-output"

    with pytest.raises(ValueError, match="parent must be an existing directory"):
        _validate_output_target(output_dir)

    assert not output_dir.parent.exists()


def test_publish_staging_refuses_to_replace_a_concurrently_created_target(tmp_path: Path) -> None:
    staging = tmp_path / ".q2-staging"
    staging.mkdir()
    (staging / "ours.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "q2-output"
    target.mkdir()
    protected = target / "other-run.txt"
    protected.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _publish_staging(staging, target)

    assert protected.read_text(encoding="utf-8") == "keep"
    assert staging.is_dir()
