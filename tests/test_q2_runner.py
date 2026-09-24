from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import replace
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
from e_mosei_audit.q2.missingness import FeatureNormalizer
from e_mosei_audit.q2.model import MaskAwareTemporalFusion, Q2Output
from e_mosei_audit.q2.runner import (
    Prediction,
    _joint_loss,
    _publish_staging,
    _validate_output_target,
    check_q2,
    classification_report,
    compute_metrics,
    evaluate_saved_q2_valid,
    run_q2,
    write_predictions,
)
import e_mosei_audit.q2.runner as q2_runner


def test_joint_loss_applies_configured_regression_weight() -> None:
    output = Q2Output(
        logits=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
        score=torch.tensor([0.5, -0.25]),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
    )
    labels = torch.tensor([0, 1])
    scores = torch.tensor([0.0, -1.0])
    class_weights = torch.tensor([1.0, 2.0, 1.0])
    expected = torch.nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
    expected += 0.25 * torch.nn.functional.smooth_l1_loss(output.score, scores)

    actual = _joint_loss(
        output,
        labels,
        scores,
        class_weights,
        regression_loss_weight=0.25,
        polarity_consistency_loss_weight=0.0,
    )

    assert torch.allclose(actual, expected)


def test_joint_loss_applies_configured_polarity_consistency_weight() -> None:
    output = Q2Output(
        logits=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
        score=torch.tensor([0.5, -0.25]),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
    )
    labels = torch.tensor([0, 1])
    scores = torch.tensor([0.0, -1.0])
    class_weights = torch.tensor([1.0, 2.0, 1.0])
    expected = torch.nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
    expected += 0.25 * torch.nn.functional.smooth_l1_loss(output.score, scores)
    probabilities = torch.softmax(output.logits, dim=1)
    expected_polarity = probabilities @ torch.tensor([-1.0, 0.0, 1.0])
    expected += 0.10 * torch.nn.functional.smooth_l1_loss(output.score / 3.0, expected_polarity)

    actual = _joint_loss(
        output,
        labels,
        scores,
        class_weights,
        regression_loss_weight=0.25,
        polarity_consistency_loss_weight=0.10,
    )

    assert torch.allclose(actual, expected)


def test_joint_loss_polarity_consistency_term_backpropagates_to_both_heads(monkeypatch) -> None:
    output = Q2Output(
        logits=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], requires_grad=True),
        score=torch.tensor([0.0, 0.0], requires_grad=True),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
    )
    monkeypatch.setattr(
        q2_runner.nn.functional,
        "cross_entropy",
        lambda logits, labels, weight: logits.sum() * 0.0,
    )

    loss = q2_runner._joint_loss(
        output,
        torch.tensor([0, 1]),
        torch.tensor([0.0, 0.0]),
        torch.ones(3),
        regression_loss_weight=0.0,
        polarity_consistency_loss_weight=1.0,
    )
    loss.backward()

    assert output.logits.grad is not None
    assert torch.isfinite(output.logits.grad).all()
    assert torch.count_nonzero(output.logits.grad) > 0
    assert output.score.grad is not None
    assert torch.isfinite(output.score.grad).all()
    assert torch.count_nonzero(output.score.grad) > 0


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


def test_classification_report_returns_confusion_matrix_and_per_class_metrics() -> None:
    report = classification_report(
        true_classes=np.array([0, 0, 0, 1, 1, 2]),
        predicted_classes=np.array([0, 0, 1, 1, 2, 2]),
    )

    assert report["confusion_matrix"] == {
        "rows": "true_class",
        "columns": "predicted_class",
        "labels": ["Negative", "Neutral", "Positive"],
        "counts": [[2, 1, 0], [0, 1, 1], [0, 0, 1]],
    }
    assert report["per_class"] == {
        "Negative": {"label": 0, "support": 3, "precision": 1.0, "recall": 2 / 3, "f1": 0.8},
        "Neutral": {"label": 1, "support": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        "Positive": {"label": 2, "support": 1, "precision": 0.5, "recall": 1.0, "f1": 2 / 3},
    }


def test_classification_report_rejects_classes_outside_the_fixed_three_labels() -> None:
    with pytest.raises(ValueError, match="classes must only contain 0, 1, or 2"):
        classification_report(
            true_classes=np.array([0, 3]),
            predicted_classes=np.array([0, 2]),
        )


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


class TrainValidPayloadWithInaccessibleTest(dict[str, object]):
    """Expose only train and valid when a valid-only flow reads an aligned payload."""

    def __getitem__(self, key: str) -> object:
        if key == "test":
            raise AssertionError("Attachment 2 test split must not be accessed")
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:
        if key == "test":
            raise AssertionError("Attachment 2 test split must not be accessed")
        return super().get(key, default)


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
        regression_loss_weight=0.5,
        polarity_consistency_loss_weight=0.0,
        class_weight_exponent=1.0,
        synthetic_missingness_enabled=True,
        fusion_variant="gated",
        text_adapter_variant="identity",
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
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    classification = json.loads((config.output_dir / "valid_classification_report.json").read_text(encoding="utf-8"))
    report = (config.output_dir / "audit_report.md").read_text(encoding="utf-8")
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert len(scenarios) == 27
    assert set(metrics["clean"]) == {"accuracy", "macro_f1", "mae", "pearson"}
    assert manifest["training"]["regression_loss_weight"] == 0.5
    assert manifest["training"]["polarity_consistency_loss_weight"] == 0.0
    assert manifest["training"]["class_weight_exponent"] == 1.0
    assert manifest["training"]["fusion_variant"] == "gated"
    assert manifest["training"]["text_adapter_variant"] == "identity"
    assert manifest["training"]["synthetic_missingness"]["enabled"] is True
    assert classification["confusion_matrix"]["labels"] == ["Negative", "Neutral", "Positive"]
    assert sum(sum(row) for row in classification["confusion_matrix"]["counts"]) == 3
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


def test_run_q2_records_mag_lite_fusion_variant(tmp_path: Path) -> None:
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
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-mag-output",
        fusion_variant="mag_lite",
    )

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["fusion_variant"] == "mag_lite"


def test_run_q2_records_houlsby_text_adapter_variant_with_gated_fusion(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    }
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-text-adapter-output",
        fusion_variant="gated",
        text_adapter_variant="houlsby_output_b32",
    )

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["fusion_variant"] == "gated"
    assert manifest["training"]["text_adapter_variant"] == "houlsby_output_b32"


def test_run_q2_records_mult_lite_fusion_variant(tmp_path: Path) -> None:
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
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-mult-output",
        fusion_variant="mult_lite",
    )

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["fusion_variant"] == "mult_lite"


def test_run_q2_records_shared_late_expert_fusion_without_accessing_test(tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload["test"]
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload.get("test")
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: aligned_payload,
    }
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-late-expert-output",
        fusion_variant="late_expert_shared",
        text_adapter_variant="identity",
    )

    summary = run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["fusion_variant"] == "late_expert_shared"
    assert manifest["training"]["text_adapter_variant"] == "identity"


def test_run_q2_forwards_configured_regression_loss_weight_to_training(monkeypatch, tmp_path: Path) -> None:
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
    observed_weights: list[float] = []
    original_joint_loss = q2_runner._joint_loss

    def recording_joint_loss(*args, **kwargs):
        observed_weights.append(kwargs["regression_loss_weight"])
        return original_joint_loss(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "_joint_loss", recording_joint_loss)
    config = replace(runner_config(tmp_path), regression_loss_weight=0.25)

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    assert observed_weights == [0.25]


def test_run_q2_forwards_configured_polarity_consistency_loss_weight_to_training(monkeypatch, tmp_path: Path) -> None:
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
    observed_weights: list[float] = []
    original_joint_loss = q2_runner._joint_loss

    def recording_joint_loss(*args, **kwargs):
        observed_weights.append(kwargs["polarity_consistency_loss_weight"])
        return original_joint_loss(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "_joint_loss", recording_joint_loss)
    config = replace(runner_config(tmp_path), polarity_consistency_loss_weight=0.37)

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    assert observed_weights == [0.37]
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["polarity_consistency_loss_weight"] == 0.37


def test_run_q2_skips_training_synthetic_missingness_when_disabled(monkeypatch, tmp_path: Path) -> None:
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

    def fail_if_called(*args, **kwargs):
        raise AssertionError("training synthetic missingness must be disabled")

    monkeypatch.setattr(q2_runner, "apply_contiguous_drop", fail_if_called)
    config = replace(runner_config(tmp_path), synthetic_missingness_enabled=False)

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["synthetic_missingness"]["enabled"] is False


def test_run_q2_applies_training_synthetic_missingness_when_enabled(monkeypatch, tmp_path: Path) -> None:
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
    invocations: list[tuple[object, object]] = []
    original_apply_contiguous_drop = q2_runner.apply_contiguous_drop

    def recording_apply_contiguous_drop(*args, **kwargs):
        invocations.append((args, kwargs))
        return original_apply_contiguous_drop(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "apply_contiguous_drop", recording_apply_contiguous_drop)
    config = replace(runner_config(tmp_path), synthetic_missingness_enabled=True)

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert invocations
    assert manifest["training"]["synthetic_missingness"]["enabled"] is True


def test_class_weights_use_normalized_inverse_frequency_exponent() -> None:
    labels = np.array([0, 0, 1, 2], dtype=np.int64)

    baseline = q2_runner._class_weights(labels, torch.device("cpu"), exponent=1.0)
    exponent_two = q2_runner._class_weights(labels, torch.device("cpu"), exponent=2.0)

    assert torch.allclose(baseline, torch.tensor([2 / 3, 4 / 3, 4 / 3]))
    assert torch.allclose(exponent_two, torch.tensor([0.4, 1.6, 1.6]))
    assert torch.isclose(exponent_two[torch.as_tensor(labels)].mean(), torch.tensor(1.0))


def test_class_weights_baseline_exponent_preserves_legacy_float32_formula() -> None:
    labels = np.repeat(np.arange(3, dtype=np.int64), [7, 11, 13])
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    expected = torch.as_tensor(counts.sum() / (3.0 * counts), device=torch.device("cpu"))

    actual = q2_runner._class_weights(labels, torch.device("cpu"), exponent=1.0)

    assert torch.equal(actual, expected)


def test_class_weights_extreme_exponent_remains_finite_for_balanced_classes() -> None:
    labels = np.repeat(np.arange(3, dtype=np.int64), 2)

    weights = q2_runner._class_weights(labels, torch.device("cpu"), exponent=150.0)

    assert torch.isfinite(weights).all()
    assert torch.equal(weights, torch.ones(3))
    assert torch.equal(weights[torch.as_tensor(labels)].mean(), torch.tensor(1.0))


def test_run_q2_forwards_configured_class_weight_exponent_to_training(monkeypatch, tmp_path: Path) -> None:
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
    observed_exponents: list[float] = []
    original_class_weights = q2_runner._class_weights

    def recording_class_weights(*args, **kwargs):
        observed_exponents.append(kwargs["exponent"])
        return original_class_weights(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "_class_weights", recording_class_weights)
    config = replace(runner_config(tmp_path), class_weight_exponent=1.25)

    run_q2(config, archive=RunnerArchive(members), token_encoder=TinyTokenEncoder())

    assert observed_exponents == [1.25]


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


def test_evaluate_saved_q2_valid_reconstructs_legacy_gated_identity_without_accessing_test(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "valid-report.json"

    report = evaluate_saved_q2_valid(
        run_dir,
        output,
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["sample_count"] == 3
    assert sum(sum(row) for row in report["confusion_matrix"]["counts"]) == 3
    assert set(report["prediction_score_summary"]) == {"mean", "std", "min", "max"}
    assert archive.verify_count == 1


def test_evaluate_saved_q2_valid_reconstructs_houlsby_text_adapter_checkpoint(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-text-adapter-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="gated",
        text_adapter_variant="houlsby_output_b32",
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "gated",
                    "text_adapter_variant": "houlsby_output_b32",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "text-adapter-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1


def test_evaluate_saved_q2_valid_reconstructs_mag_lite_checkpoint(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-mag-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mag_lite"
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "mag_lite",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "mag-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1


def test_evaluate_saved_q2_valid_reconstructs_mult_lite_checkpoint(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-mult-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mult_lite"
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "mult_lite",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "mult-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1


def test_evaluate_saved_q2_valid_strictly_reconstructs_shared_late_expert_checkpoint(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload["test"]
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload.get("test")
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: aligned_payload,
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-late-expert-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "late_expert_shared",
                    "text_adapter_variant": "identity",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "late-expert-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert sum(sum(row) for row in report["confusion_matrix"]["counts"]) == 3
    assert archive.verify_count == 1


def test_evaluate_saved_q2_valid_rejects_unsupported_manifest_fusion_variant(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-variant-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "unsupported",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Afusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared\Z",
    ):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-variant-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


def test_evaluate_saved_q2_valid_rejects_unsupported_manifest_text_adapter_variant(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-text-adapter-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "fusion_variant": "gated",
                    "text_adapter_variant": "unsupported",
                    "device": "cpu",
                },
                "normalizer": FeatureNormalizer(
                    audio_mean=np.zeros(74, dtype=np.float32),
                    audio_std=np.ones(74, dtype=np.float32),
                    vision_mean=np.zeros(35, dtype=np.float32),
                    vision_std=np.ones(35, dtype=np.float32),
                ).as_dict(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="text_adapter_variant must be one of: identity, houlsby_output_b32",
    ):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-text-adapter-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


def test_evaluate_saved_q2_valid_rejects_manifest_with_missing_normalizer_field(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": {
                    "batch_size": 3,
                    "hidden_size": 16,
                    "heads": 4,
                    "layers": 1,
                    "dropout": 0.0,
                    "device": "cpu",
                },
                "normalizer": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run manifest normalizer missing audio_mean"):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "valid-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


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
