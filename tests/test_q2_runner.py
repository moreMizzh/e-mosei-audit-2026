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


class RecordingOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


def flat_output(logits: torch.Tensor, score: torch.Tensor) -> Q2Output:
    return Q2Output(
        logits=logits,
        score=score,
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
    )


def test_classification_loss_uses_weighted_label_smoothing_and_backpropagates() -> None:
    logits = torch.tensor([[1.2, -0.3, 0.4], [-0.8, 0.9, 0.1]], requires_grad=True)
    labels = torch.tensor([0, 2])
    class_weights = torch.tensor([0.2, 3.0, 1.7])
    output = flat_output(logits, torch.zeros(2))

    actual = q2_runner._classification_loss(
        output,
        labels,
        class_weights,
        classification_loss_variant="weighted_label_smoothing_005",
    )
    expected = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=0.05,
    )

    assert torch.equal(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


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
        classification_loss_variant="hard_ce",
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
        classification_loss_variant="hard_ce",
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
        classification_loss_variant="hard_ce",
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


def test_joint_loss_uses_weighted_conditional_bce_for_corn() -> None:
    ordinal_logits = torch.tensor([[0.2, -0.4], [0.7, 1.1], [-0.3, 0.9]])
    output = Q2Output(
        logits=torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]]).repeat(3, 1), dim=1),
        score=torch.zeros(3),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
        ordinal_logits=ordinal_logits,
    )
    labels = torch.tensor([0, 1, 2])
    scores = torch.zeros(3)
    class_weights = torch.tensor([0.1, 7.0, 0.3])
    sample_weights = class_weights[labels]
    first_terms = torch.nn.functional.binary_cross_entropy_with_logits(
        ordinal_logits[:, 0], torch.tensor([0.0, 1.0, 1.0]), reduction="none"
    )
    first = (first_terms * sample_weights).sum() / sample_weights.sum()
    active = labels > 0
    second_terms = torch.nn.functional.binary_cross_entropy_with_logits(
        ordinal_logits[active, 1], torch.tensor([0.0, 1.0]), reduction="none"
    )
    second = (second_terms * sample_weights[active]).sum() / sample_weights[active].sum()
    expected = torch.stack((first, second)).mean()

    actual = _joint_loss(
        output,
        labels,
        scores,
        class_weights,
        classification_loss_variant="hard_ce",
        regression_loss_weight=0.0,
        polarity_consistency_loss_weight=0.0,
    )
    assert torch.allclose(actual, expected)


def test_joint_loss_corn_all_negative_batch_uses_only_first_condition() -> None:
    ordinal_logits = torch.tensor([[0.2, -0.4], [-0.7, 1.1]])
    output = Q2Output(
        logits=torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]]).repeat(2, 1), dim=1),
        score=torch.zeros(2),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
        ordinal_logits=ordinal_logits,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(ordinal_logits[:, 0], torch.zeros(2))

    actual = _joint_loss(
        output,
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2),
        torch.tensor([3.0, 2.0, 1.0]),
        classification_loss_variant="hard_ce",
        regression_loss_weight=0.0,
        polarity_consistency_loss_weight=0.0,
    )

    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected)


def test_rdrop_joint_loss_matches_two_weighted_joint_losses_and_symmetric_kl() -> None:
    view_one = flat_output(
        torch.tensor([[1.0, -0.5, 0.25], [-0.25, 0.5, 1.0]], requires_grad=True),
        torch.tensor([0.75, -0.5], requires_grad=True),
    )
    view_two = flat_output(
        torch.tensor([[0.25, 0.75, -1.0], [1.25, -0.5, 0.0]], requires_grad=True),
        torch.tensor([-0.25, 0.5], requires_grad=True),
    )
    labels = torch.tensor([0, 2])
    scores = torch.tensor([0.0, -1.0])
    class_weights = torch.tensor([1.0, 2.0, 1.5])
    expected = 0.5 * (
        _joint_loss(
            view_one,
            labels,
            scores,
            class_weights,
            classification_loss_variant="hard_ce",
            regression_loss_weight=0.25,
            polarity_consistency_loss_weight=0.10,
        )
        + _joint_loss(
            view_two,
            labels,
            scores,
            class_weights,
            classification_loss_variant="hard_ce",
            regression_loss_weight=0.25,
            polarity_consistency_loss_weight=0.10,
        )
    )
    expected += 0.25 * (
        torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(view_one.logits, dim=1),
            torch.nn.functional.softmax(view_two.logits, dim=1),
            reduction="batchmean",
        )
        + torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(view_two.logits, dim=1),
            torch.nn.functional.softmax(view_one.logits, dim=1),
            reduction="batchmean",
        )
    )

    actual = q2_runner._rdrop_joint_loss(
        view_one,
        view_two,
        labels,
        scores,
        class_weights,
        classification_loss_variant="hard_ce",
        regression_loss_weight=0.25,
        polarity_consistency_loss_weight=0.10,
    )

    assert torch.allclose(actual, expected)
    actual.backward()
    for tensor in (view_one.logits, view_one.score, view_two.logits, view_two.score):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_train_epoch_none_uses_one_forward_and_joint_loss(monkeypatch) -> None:
    output = flat_output(
        torch.tensor([[0.25, 0.5, -0.75]], requires_grad=True), torch.tensor([0.25], requires_grad=True)
    )
    labels = torch.tensor([1])
    scores = torch.tensor([0.0])
    calls: list[tuple[object, object, object, object, object]] = []
    joint_outputs: list[Q2Output] = []
    optimizer = RecordingOptimizer()
    split = type("OneSampleSplit", (), {"sample_count": 1})()
    batch_masks = object()

    def fake_forward(model, encoder, passed_split, indexes, dropped, normalizer, device):
        calls.append((passed_split, indexes, dropped, normalizer, device))
        return output, labels, scores

    def recording_joint_loss(passed_output, *args, **kwargs):
        joint_outputs.append(passed_output)
        return passed_output.logits.sum() + passed_output.score.sum()

    monkeypatch.setattr(q2_runner, "_batch_indexes", lambda *args: [np.asarray([0])])
    monkeypatch.setattr(q2_runner, "_slice_masks", lambda *args: batch_masks)
    monkeypatch.setattr(q2_runner, "_forward_split", fake_forward)
    monkeypatch.setattr(q2_runner, "_joint_loss", recording_joint_loss)

    q2_runner._train_epoch(
        torch.nn.Identity(),
        object(),
        optimizer,
        split,
        object(),
        object(),
        batch_size=1,
        class_weights=torch.ones(3),
        classification_loss_variant="hard_ce",
        regression_loss_weight=0.5,
        polarity_consistency_loss_weight=0.0,
        dropout_consistency_variant="none",
        synthetic_missingness_enabled=False,
        rng=np.random.default_rng(7),
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    assert joint_outputs == [output]
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1


def test_train_epoch_rdrop_uses_two_forwards_with_one_mask_and_step(monkeypatch) -> None:
    outputs = iter(
        (
            flat_output(
                torch.tensor([[0.25, 0.5, -0.75]], requires_grad=True), torch.tensor([0.25], requires_grad=True)
            ),
            flat_output(
                torch.tensor([[-0.5, 0.25, 0.75]], requires_grad=True), torch.tensor([-0.5], requires_grad=True)
            ),
        )
    )
    labels = torch.tensor([1])
    scores = torch.tensor([0.0])
    calls: list[tuple[object, object, object, object, object]] = []
    optimizer = RecordingOptimizer()
    split = type("OneSampleSplit", (), {"sample_count": 1})()
    batch_masks = object()

    def fake_forward(model, encoder, passed_split, indexes, dropped, normalizer, device):
        calls.append((passed_split, indexes, dropped, normalizer, device))
        return next(outputs), labels, scores

    monkeypatch.setattr(q2_runner, "_batch_indexes", lambda *args: [np.asarray([0])])
    monkeypatch.setattr(q2_runner, "_slice_masks", lambda *args: batch_masks)
    monkeypatch.setattr(q2_runner, "_forward_split", fake_forward)

    q2_runner._train_epoch(
        torch.nn.Identity(),
        object(),
        optimizer,
        split,
        object(),
        object(),
        batch_size=1,
        class_weights=torch.ones(3),
        classification_loss_variant="hard_ce",
        regression_loss_weight=0.5,
        polarity_consistency_loss_weight=0.0,
        dropout_consistency_variant="rdrop_alpha_1",
        synthetic_missingness_enabled=False,
        rng=np.random.default_rng(7),
        device=torch.device("cpu"),
    )

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] is calls[1][1]
    assert calls[0][2] is calls[1][2]
    assert calls[0][3] is calls[1][3]
    assert calls[0][4] is calls[1][4]
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1


def test_train_epoch_rdrop_rejects_ordinal_outputs_before_optimizer_mutation(monkeypatch) -> None:
    ordinal_output = Q2Output(
        logits=torch.tensor([[0.25, 0.5, -0.75]], requires_grad=True),
        score=torch.tensor([0.25], requires_grad=True),
        gates=torch.empty(0),
        temporal_attention=torch.empty(0),
        ordinal_logits=torch.zeros((1, 2)),
    )
    labels = torch.tensor([1])
    scores = torch.tensor([0.0])
    forward_calls = 0
    optimizer = RecordingOptimizer()
    split = type("OneSampleSplit", (), {"sample_count": 1})()

    def fake_forward(*args):
        nonlocal forward_calls
        forward_calls += 1
        return ordinal_output, labels, scores

    monkeypatch.setattr(q2_runner, "_batch_indexes", lambda *args: [np.asarray([0])])
    monkeypatch.setattr(q2_runner, "_slice_masks", lambda *args: object())
    monkeypatch.setattr(q2_runner, "_forward_split", fake_forward)

    with pytest.raises(ValueError, match=r"\AR-Drop requires flat classification outputs\Z"):
        q2_runner._train_epoch(
            torch.nn.Identity(),
            object(),
            optimizer,
            split,
            object(),
            object(),
            batch_size=1,
            class_weights=torch.ones(3),
            classification_loss_variant="hard_ce",
            regression_loss_weight=0.5,
            polarity_consistency_loss_weight=0.0,
            dropout_consistency_variant="rdrop_alpha_1",
            synthetic_missingness_enabled=False,
            rng=np.random.default_rng(7),
            device=torch.device("cpu"),
        )

    assert forward_calls == 2
    assert optimizer.zero_grad_calls == 0
    assert optimizer.step_calls == 0


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


class ArchiveAccessSentinel:
    """Fail if runner-entry validation attempts to touch an archive."""

    def __init__(self) -> None:
        object.__setattr__(self, "accesses", [])

    def __getattribute__(self, name: str) -> object:
        object.__getattribute__(self, "accesses").append(name)
        raise AssertionError(f"archive must not be accessed: {name}")


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


class ScalarMixTokenEncoder:
    """Small trainable stand-in for the frozen scalar-mix encoder boundary."""

    def __init__(self) -> None:
        self.layer_logits = torch.nn.Parameter(torch.zeros(4))
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.loaded_states: list[dict[str, torch.Tensor]] = []

    def encode(self, token_rows: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0)
        multiplier = self.scale * (weights * torch.arange(1, 5, device=token_rows.device)).sum()
        return multiplier * token_rows[:, 0, :].to(torch.float32).unsqueeze(-1).expand(-1, -1, 768)

    def trainable_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (self.layer_logits, self.scale)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "layer_logits": self.layer_logits.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
        }

    def load_trainable_state_dict(self, state: object) -> None:
        if not isinstance(state, dict) or set(state) != {"layer_logits", "scale"}:
            raise ValueError("scalar-mix trainable state must contain exactly: layer_logits, scale")
        expected = {"layer_logits": (4,), "scale": ()}
        for name, shape in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"scalar-mix state {name} must be a finite tensor with shape {shape}")
        restored = {name: state[name].detach().cpu().clone() for name in expected}
        self.loaded_states.append(restored)
        with torch.no_grad():
            self.layer_logits.copy_(state["layer_logits"])
            self.scale.copy_(state["scale"])


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


def runner_config(tmp_path: Path, *, temporal_pooling_variant: str = "attention") -> Q2Config:
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
        dropout_consistency_variant="none",
        class_weight_exponent=1.0,
        synthetic_missingness_enabled=True,
        fusion_variant="gated",
        text_adapter_variant="identity",
        classification_variant="flat",
        classification_loss_variant="hard_ce",
        temporal_position_variant="none",
        temporal_context_variant="none",
        temporal_residual_variant="none",
        temporal_pooling_variant=temporal_pooling_variant,
        text_encoder_variant="last_hidden_state",
        device="cpu",
    )


def _write_saved_temporal_context_run(
    run_dir: Path,
    config: Q2Config,
    *,
    temporal_context_variant: str | None,
    fusion_variant: str = "gated",
) -> None:
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant=fusion_variant,
        temporal_context_variant=temporal_context_variant or "none",
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    training: dict[str, object] = {
        "batch_size": 3,
        "hidden_size": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.0,
        "fusion_variant": fusion_variant,
        "text_adapter_variant": "identity",
        "classification_variant": "flat",
        "temporal_position_variant": "none",
        "temporal_pooling_variant": "attention",
        "device": "cpu",
    }
    if temporal_context_variant is not None:
        training["temporal_context_variant"] = temporal_context_variant
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "archive": str(config.archive),
                "seven_zip": str(config.seven_zip),
                "bert_model": str(config.bert_model),
                "training": training,
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


def runner_archive_with_inaccessible_test() -> RunnerArchive:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    return RunnerArchive(members)


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
    assert "architecture" not in manifest
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
    assert not (config.output_dir / "text_encoder_state.pt").exists()
    assert archive.verify_count == 1
    assert "缺失影响汇总" in report
    assert "最不利 macro-F1 场景" in report


def test_run_q2_records_weighted_label_smoothing_without_accessing_test(tmp_path: Path) -> None:
    config = replace(
        runner_config(tmp_path),
        classification_loss_variant="weighted_label_smoothing_005",
    )

    summary = run_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))

    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["classification_loss_variant"] == "weighted_label_smoothing_005"


def test_rdrop_run_manifest_reconstructs_valid_without_test_or_attachment3(
    monkeypatch, tmp_path: Path
) -> None:
    config = replace(
        runner_config(tmp_path),
        dropout=0.1,
        dropout_consistency_variant="rdrop_alpha_1",
    )

    summary = run_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    valid_only_archive = RunnerArchive(
        {
            ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                {
                    "train": runner_split([0, 1, 2]),
                    "valid": runner_split([0, 1, 2]),
                    "test": {"not": "a valid evaluation input"},
                }
            )
        }
    )
    observed_strict: list[bool] = []
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)
    report = evaluate_saved_q2_valid(
        config.output_dir,
        tmp_path / "rdrop-valid-report.json",
        archive=valid_only_archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["dropout_consistency_variant"] == "rdrop_alpha_1"
    assert report["sample_count"] == 3
    assert valid_only_archive.verify_count == 1
    assert observed_strict == [True]


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


def test_run_q2_records_pairwise_hadamard_residual_without_test_split_access(monkeypatch, tmp_path: Path) -> None:
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
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-pairwise-hadamard-output",
        fusion_variant="pairwise_hadamard_residual",
    )
    observed_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion

    def recording_model_constructor(*args, **kwargs):
        observed_variants.append(kwargs["fusion_variant"])
        return original_model_constructor(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["fusion_variant"] == "pairwise_hadamard_residual"
    assert archive.verify_count == 1
    assert observed_variants == ["pairwise_hadamard_residual"]


def test_run_q2_records_sinusoidal_temporal_position_without_test_split_access(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-sinusoidal-position-output",
        temporal_position_variant="sinusoidal",
    )
    observed_position_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion

    def recording_model_constructor(*args, **kwargs):
        observed_position_variants.append(kwargs["temporal_position_variant"])
        return original_model_constructor(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["temporal_position_variant"] == "sinusoidal"
    assert archive.verify_count == 1
    assert observed_position_variants == ["sinusoidal"]


def test_run_q2_records_attention_availability_pooling_without_test_split_access(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-attention-availability-output",
        temporal_pooling_variant="attention_availability",
    )
    observed_pooling_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion

    def recording_model_constructor(*args, **kwargs):
        observed_pooling_variants.append(kwargs["temporal_pooling_variant"])
        return original_model_constructor(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["temporal_pooling_variant"] == "attention_availability"
    assert archive.verify_count == 1
    assert observed_pooling_variants == ["attention_availability"]


def test_masked_mean_config_runs_check_and_fake_archive_without_accessing_test(tmp_path: Path) -> None:
    config = runner_config(tmp_path, temporal_pooling_variant="masked_mean")

    check_summary = check_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )
    assert not config.output_dir.exists()
    run_summary = run_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert check_summary == {"train_count": 3, "valid_count": 3, "attachment3_count": 30}
    assert config.output_dir.is_dir()
    assert run_summary["attachment3_count"] == 30
    assert manifest["training"]["temporal_pooling_variant"] == "masked_mean"


def test_availability_context_runs_check_and_fake_archive_without_accessing_test(monkeypatch, tmp_path: Path) -> None:
    config = replace(runner_config(tmp_path), temporal_context_variant="availability_embedding")
    observed_context_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion

    def recording_model_constructor(*args, **kwargs):
        observed_context_variants.append(kwargs["temporal_context_variant"])
        return original_model_constructor(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)

    check_summary = check_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )
    assert not config.output_dir.exists()
    run_summary = run_q2(
        config,
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert check_summary == {"train_count": 3, "valid_count": 3, "attachment3_count": 30}
    assert run_summary["attachment3_count"] == 30
    assert manifest["training"]["temporal_context_variant"] == "availability_embedding"
    assert observed_context_variants == ["availability_embedding"]


@pytest.mark.parametrize("entrypoint", [run_q2, check_q2], ids=["run", "check"])
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"temporal_context_variant": "unsupported"},
            "temporal_context_variant must be one of: none, availability_embedding",
        ),
        (
            {
                "temporal_context_variant": "availability_embedding",
                "fusion_variant": "late_expert_shared",
            },
            "availability_embedding temporal context is unsupported with late_expert_shared fusion",
        ),
    ],
)
def test_q2_entrypoints_reject_invalid_temporal_context_before_output_or_archive(
    tmp_path: Path,
    entrypoint,
    updates: dict[str, object],
    message: str,
) -> None:
    config = replace(runner_config(tmp_path), **updates)
    archive = ArchiveAccessSentinel()

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        entrypoint(config, archive=archive, token_encoder=TinyTokenEncoder())

    assert object.__getattribute__(archive, "accesses") == []
    assert not config.output_dir.exists()


def test_run_q2_records_pooled_lmf_r4_architecture_without_test_split_access(tmp_path: Path) -> None:
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
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-pooled-lmf-output",
        fusion_variant="pooled_lmf_r4",
    )

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["fusion_variant"] == "pooled_lmf_r4"
    assert manifest["architecture"] == {"pooled_lmf_rank": 4}
    assert archive.verify_count == 1


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


def test_run_q2_records_text_anchor_residual_fusion_without_accessing_test(tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-text-anchor-output",
        fusion_variant="text_anchor_residual",
    )

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["fusion_variant"] == "text_anchor_residual"
    assert archive.verify_count == 1


def test_run_q2_records_corn_variant_without_accessing_test(tmp_path: Path) -> None:
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
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    for index in range(1, 31):
        path = f"E题数据/附件3-模态缺失特征样本/对齐版本/附件3_{index:02d}.pkl"
        members[path] = runner_attachment3_payload(index)
    archive = RunnerArchive(members)
    config = replace(
        runner_config(tmp_path),
        output_dir=tmp_path / "q2-corn-output",
        classification_variant="corn",
    )

    summary = run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert summary["attachment3_count"] == 30
    assert len(predictions) == 30
    assert manifest["training"]["classification_variant"] == "corn"
    assert archive.verify_count == 1


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


def test_run_q2_forwards_dropout_consistency_variant_to_training(monkeypatch, tmp_path: Path) -> None:
    observed_variants: list[str] = []
    original_train_epoch = q2_runner._train_epoch

    def recording_train_epoch(*args, **kwargs):
        observed_variants.append(kwargs["dropout_consistency_variant"])
        return original_train_epoch(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "_train_epoch", recording_train_epoch)
    config = replace(
        runner_config(tmp_path),
        dropout=0.1,
        dropout_consistency_variant="rdrop_alpha_1",
    )

    run_q2(config, archive=runner_archive_with_inaccessible_test(), token_encoder=TinyTokenEncoder())

    assert observed_variants == ["rdrop_alpha_1"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"dropout_consistency_variant": "unsupported"},
            "dropout_consistency_variant must be one of: none, rdrop_alpha_1",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": 0.0},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": float("nan")},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": float("inf")},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {
                "dropout_consistency_variant": "rdrop_alpha_1",
                "dropout": 0.1,
                "classification_variant": "corn",
            },
            "rdrop_alpha_1 requires classification_variant=flat",
        ),
    ],
)
def test_run_q2_rejects_direct_invalid_dropout_semantic_before_output_or_archive(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    config = replace(runner_config(tmp_path), **updates)
    archive = runner_archive_with_inaccessible_test()

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        run_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    assert archive.verify_count == 0
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "classification_loss_variant": "weighted_label_smoothing_005",
                "dropout_consistency_variant": "rdrop_alpha_1",
                "dropout": 0.1,
            },
            "weighted_label_smoothing_005 cannot be combined with rdrop_alpha_1",
        ),
        (
            {
                "classification_loss_variant": "weighted_label_smoothing_005",
                "classification_variant": "corn",
            },
            "weighted_label_smoothing_005 requires classification_variant=flat",
        ),
    ],
)
def test_run_q2_rejects_direct_invalid_classification_loss_semantic_before_archive(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    config = replace(runner_config(tmp_path), **updates)

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        run_q2(config, archive=ArchiveAccessSentinel(), token_encoder=TinyTokenEncoder())

    assert not config.output_dir.exists()


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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"dropout_consistency_variant": "unsupported"},
            "dropout_consistency_variant must be one of: none, rdrop_alpha_1",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": 0.0},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": float("nan")},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {"dropout_consistency_variant": "rdrop_alpha_1", "dropout": float("inf")},
            "rdrop_alpha_1 requires a finite dropout > 0",
        ),
        (
            {
                "dropout_consistency_variant": "rdrop_alpha_1",
                "dropout": 0.1,
                "classification_variant": "corn",
            },
            "rdrop_alpha_1 requires classification_variant=flat",
        ),
    ],
)
def test_check_q2_rejects_direct_invalid_dropout_semantic_before_output_or_archive(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    config = replace(runner_config(tmp_path), **updates)
    archive = runner_archive_with_inaccessible_test()

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        check_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    assert archive.verify_count == 0
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "classification_loss_variant": "weighted_label_smoothing_005",
                "classification_variant": "corn",
            },
            "weighted_label_smoothing_005 requires classification_variant=flat",
        ),
        (
            {
                "classification_loss_variant": "weighted_label_smoothing_005",
                "dropout_consistency_variant": "rdrop_alpha_1",
                "dropout": 0.1,
            },
            "weighted_label_smoothing_005 cannot be combined with rdrop_alpha_1",
        ),
    ],
)
def test_check_q2_rejects_direct_invalid_classification_loss_semantic_before_output_or_archive(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    config = replace(runner_config(tmp_path), **updates)
    archive = ArchiveAccessSentinel()

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        check_q2(config, archive=archive, token_encoder=TinyTokenEncoder())

    assert object.__getattribute__(archive, "accesses") == []
    assert not config.output_dir.exists()


def test_manifest_dropout_consistency_variant_defaults_and_validates() -> None:
    assert q2_runner._manifest_dropout_consistency_variant({}) == "none"
    assert q2_runner._manifest_dropout_consistency_variant({"dropout_consistency_variant": "none"}) == "none"
    assert (
        q2_runner._manifest_dropout_consistency_variant(
            {
                "dropout_consistency_variant": "rdrop_alpha_1",
                "classification_variant": "flat",
                "dropout": 0.1,
            }
        )
        == "rdrop_alpha_1"
    )
    with pytest.raises(
        ValueError,
        match=r"\Adropout_consistency_variant must be one of: none, rdrop_alpha_1\Z",
    ):
        q2_runner._manifest_dropout_consistency_variant({"dropout_consistency_variant": "unsupported"})


def test_manifest_classification_loss_variant_defaults_and_validates() -> None:
    assert q2_runner._manifest_classification_loss_variant({}) == "hard_ce"
    assert (
        q2_runner._manifest_classification_loss_variant(
            {"classification_loss_variant": "weighted_label_smoothing_005"}
        )
        == "weighted_label_smoothing_005"
    )
    with pytest.raises(
        ValueError,
        match=r"\Aclassification_loss_variant must be one of: hard_ce, weighted_label_smoothing_005\Z",
    ):
        q2_runner._manifest_classification_loss_variant({"classification_loss_variant": "unsupported"})


def test_evaluate_saved_q2_valid_reconstructs_legacy_gated_identity_without_accessing_test(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
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
    observed_position_variants: list[str] = []
    observed_pooling_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_position_variants.append(kwargs["temporal_position_variant"])
        observed_pooling_variants.append(kwargs["temporal_pooling_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

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
    assert observed_position_variants == ["none"]
    assert observed_pooling_variants == ["attention"]
    assert observed_strict == [True]


def test_evaluate_saved_q2_valid_historical_absent_pooling_reconstructs_attention(
    monkeypatch, tmp_path: Path
) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "historical-absent-pooling-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="gated",
        classification_variant="flat",
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "none",
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
    observed_pooling_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_pooling_variants.append(kwargs["temporal_pooling_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "historical-absent-pooling-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed_pooling_variants == ["attention"]
    assert observed_strict == [True]


def test_evaluate_saved_q2_valid_strictly_reconstructs_flat_label_smoothing_checkpoint_without_accessing_test(
    monkeypatch, tmp_path: Path
) -> None:
    archive = RunnerArchive(
        {
            ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                {
                    "train": runner_split([0, 1, 2]),
                    "valid": runner_split([0, 1, 2]),
                    "test": {"not": "an accessible Q2 runtime input"},
                }
            )
        }
    )
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-label-smoothing-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, classification_variant="flat")
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
                    "classification_variant": "flat",
                    "classification_loss_variant": "weighted_label_smoothing_005",
                    "dropout_consistency_variant": "none",
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
    observed_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_variants.append(kwargs["classification_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "label-smoothing-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed_variants == ["flat"]
    assert observed_strict == [True]


@pytest.mark.parametrize(
    ("classification_loss_variant", "classification_variant", "dropout_consistency_variant", "message"),
    [
        (
            "unsupported",
            "flat",
            "none",
            "classification_loss_variant must be one of: hard_ce, weighted_label_smoothing_005",
        ),
        (
            "weighted_label_smoothing_005",
            "corn",
            "none",
            "weighted_label_smoothing_005 requires classification_variant=flat",
        ),
        (
            "weighted_label_smoothing_005",
            "flat",
            "rdrop_alpha_1",
            "weighted_label_smoothing_005 cannot be combined with rdrop_alpha_1",
        ),
    ],
)
def test_evaluate_saved_q2_valid_rejects_invalid_classification_loss_semantic_before_archive(
    tmp_path: Path,
    classification_loss_variant: str,
    classification_variant: str,
    dropout_consistency_variant: str,
    message: str,
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-classification-loss-run"
    _write_saved_scalar_mix_run(run_dir, config, text_encoder_variant="last_hidden_state")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"].update(
        {
            "classification_loss_variant": classification_loss_variant,
            "classification_variant": classification_variant,
            "dropout_consistency_variant": dropout_consistency_variant,
            "dropout": 0.1 if dropout_consistency_variant == "rdrop_alpha_1" else 0.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-classification-loss-valid-report.json",
            archive=ArchiveAccessSentinel(),
            token_encoder=TinyTokenEncoder(),
        )

    assert not (tmp_path / "invalid-classification-loss-valid-report.json").exists()


def test_evaluate_saved_q2_valid_strictly_reconstructs_attention_availability_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-attention-availability-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "none",
                    "temporal_pooling_variant": "attention_availability",
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
    observed_pooling_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_pooling_variants.append(kwargs["temporal_pooling_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "attention-availability-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed_pooling_variants == ["attention_availability"]
    assert observed_strict == [True]


def test_evaluate_saved_q2_valid_strictly_reconstructs_masked_mean_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-masked-mean-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="masked_mean",
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "none",
                    "temporal_pooling_variant": "masked_mean",
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
    observed_pooling_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_pooling_variants.append(kwargs["temporal_pooling_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "masked-mean-valid-report.json",
        archive=RunnerArchive(
            {
                ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                    {
                        "train": runner_split([0, 1, 2]),
                        "valid": runner_split([0, 1, 2]),
                        "test": {"not": "an accessible Q2 runtime input"},
                    }
                )
            }
        ),
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert observed_pooling_variants == ["masked_mean"]
    assert observed_strict == [True]


def test_evaluate_saved_q2_valid_strictly_reconstructs_availability_embedding_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-availability-embedding-run"
    _write_saved_temporal_context_run(
        run_dir,
        config,
        temporal_context_variant="availability_embedding",
    )
    observed_context_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_context_variants.append(kwargs["temporal_context_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "availability-embedding-valid-report.json",
        archive=RunnerArchive(
            {
                ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                    {
                        "train": runner_split([0, 1, 2]),
                        "valid": runner_split([0, 1, 2]),
                        "test": {"not": "an accessible Q2 runtime input"},
                    }
                )
            }
        ),
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert observed_context_variants == ["availability_embedding"]
    assert observed_strict == [True]


def test_manifest_temporal_context_variant_defaults_and_validates() -> None:
    assert q2_runner._manifest_temporal_context_variant({}) == "none"
    assert q2_runner._manifest_temporal_context_variant({"temporal_context_variant": "availability_embedding"}) == (
        "availability_embedding"
    )
    with pytest.raises(
        ValueError,
        match=r"\Atemporal_context_variant must be one of: none, availability_embedding\Z",
    ):
        q2_runner._manifest_temporal_context_variant({"temporal_context_variant": "unsupported"})


@pytest.mark.parametrize(
    ("temporal_context_variant", "fusion_variant", "message"),
    [
        (
            "unsupported",
            "gated",
            "temporal_context_variant must be one of: none, availability_embedding",
        ),
        (
            "availability_embedding",
            "late_expert_shared",
            "availability_embedding temporal context is unsupported with late_expert_shared fusion",
        ),
    ],
)
def test_evaluate_saved_q2_valid_rejects_invalid_temporal_context_before_archive(
    tmp_path: Path,
    temporal_context_variant: str,
    fusion_variant: str,
    message: str,
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-temporal-context-run"
    _write_saved_temporal_context_run(
        run_dir,
        config,
        temporal_context_variant="none",
        fusion_variant="gated",
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"]["temporal_context_variant"] = temporal_context_variant
    manifest["training"]["fusion_variant"] = fusion_variant
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    archive = ArchiveAccessSentinel()

    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-temporal-context-valid-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )

    assert object.__getattribute__(archive, "accesses") == []
    assert not (tmp_path / "invalid-temporal-context-valid-report.json").exists()


def test_evaluate_saved_q2_valid_defaults_historical_manifest_temporal_context_to_none(monkeypatch, tmp_path: Path) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-historical-temporal-context-run"
    _write_saved_temporal_context_run(run_dir, config, temporal_context_variant=None)
    observed_context_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion

    def recording_model_constructor(*args, **kwargs):
        observed_context_variants.append(kwargs["temporal_context_variant"])
        return original_model_constructor(*args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "historical-temporal-context-valid-report.json",
        archive=RunnerArchive(
            {
                ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                    {
                        "train": runner_split([0, 1, 2]),
                        "valid": runner_split([0, 1, 2]),
                        "test": {"not": "an accessible Q2 runtime input"},
                    }
                )
            }
        ),
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert observed_context_variants == ["none"]


def test_evaluate_saved_q2_valid_strictly_reconstructs_sinusoidal_position_checkpoint(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-sinusoidal-position-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "sinusoidal",
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
    observed_position_variants: list[str] = []
    observed_strict: list[bool] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_position_variants.append(kwargs["temporal_position_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "sinusoidal-position-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed_position_variants == ["sinusoidal"]
    assert observed_strict == [True]


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
    observed_strict: list[bool] = []
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
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
    assert observed_strict == [True]


def test_evaluate_saved_q2_valid_strictly_reconstructs_text_anchor_residual_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-text-anchor-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="text_anchor_residual",
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
                    "fusion_variant": "text_anchor_residual",
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
    observed_strict: list[bool] = []
    observed_fusion_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_fusion_variants.append(kwargs["fusion_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "text-anchor-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert set(report["metrics"]) == {"accuracy", "macro_f1", "mae", "pearson"}
    assert sum(sum(row) for row in report["confusion_matrix"]["counts"]) == 3
    assert archive.verify_count == 1
    assert observed_strict == [True]
    assert observed_fusion_variants == ["text_anchor_residual"]


def test_evaluate_saved_q2_valid_strictly_reconstructs_pairwise_hadamard_residual_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "an accessible Q2 runtime input"},
        }
    )
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-pairwise-hadamard-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pairwise_hadamard_residual",
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
                    "fusion_variant": "pairwise_hadamard_residual",
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
    observed_strict: list[bool] = []
    observed_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_variants.append(kwargs["fusion_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "pairwise-hadamard-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert set(report["metrics"]) == {"accuracy", "macro_f1", "mae", "pearson"}
    assert sum(sum(row) for row in report["confusion_matrix"]["counts"]) == 3
    assert archive.verify_count == 1
    assert observed_strict == [True]
    assert observed_variants == ["pairwise_hadamard_residual"]


def test_evaluate_saved_q2_valid_strictly_reconstructs_pooled_lmf_r4_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
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
    archive = RunnerArchive({ALIGNED_50_MEMBER: aligned_payload})
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-pooled-lmf-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pooled_lmf_r4",
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
                    "fusion_variant": "pooled_lmf_r4",
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
    observed_strict: list[bool] = []
    observed_variants: list[str] = []
    original_model_constructor = q2_runner.MaskAwareTemporalFusion
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_model_constructor(*args, **kwargs):
        observed_variants.append(kwargs["fusion_variant"])
        return original_model_constructor(*args, **kwargs)

    def recording_load_state_dict(self, *args, **kwargs):
        assert kwargs["strict"] is True
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", recording_model_constructor)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "pooled-lmf-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed_strict == [True]
    assert observed_variants == ["pooled_lmf_r4"]


def test_evaluate_saved_q2_valid_strictly_reconstructs_corn_checkpoint(monkeypatch, tmp_path: Path) -> None:
    aligned_payload = TrainValidPayloadWithInaccessibleTest(
        {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    )
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload["test"]
    with pytest.raises(AssertionError, match="Attachment 2 test split must not be accessed"):
        aligned_payload.get("test")
    members: dict[str, object] = {ALIGNED_50_MEMBER: aligned_payload}
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-corn-run"
    run_dir.mkdir()
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        classification_variant="corn",
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "corn",
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
    observed: list[tuple[int, bool]] = []
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    def recording_load_state_dict(self, *args, **kwargs):
        assert isinstance(self.classifier[-1], torch.nn.Linear)
        observed.append((self.classifier[-1].out_features, kwargs["strict"]))
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "corn-valid-report.json",
        archive=archive,
        token_encoder=TinyTokenEncoder(),
    )

    assert report["sample_count"] == 3
    assert archive.verify_count == 1
    assert observed == [(2, True)]


def test_evaluate_saved_q2_valid_rejects_unsupported_manifest_classification_variant(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-classification-run"
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "unsupported",
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

    with pytest.raises(ValueError, match=r"\Aclassification_variant must be one of: flat, corn\Z"):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-classification-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


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
        match=(
            r"\Afusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared, "
            r"text_anchor_residual, pairwise_hadamard_residual, pooled_lmf_r4\Z"
        ),
    ):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-variant-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


def test_evaluate_saved_q2_valid_rejects_unsupported_manifest_temporal_position_variant(tmp_path: Path) -> None:
    members: dict[str, object] = {
        ALIGNED_50_MEMBER: {
            "train": runner_split([0, 1, 2]),
            "valid": runner_split([0, 1, 2]),
            "test": {"not": "a valid evaluation input"},
        }
    }
    archive = RunnerArchive(members)
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-temporal-position-run"
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "unsupported",
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

    with pytest.raises(ValueError, match=r"\Atemporal_position_variant must be one of: none, sinusoidal\Z"):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-temporal-position-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )


def test_evaluate_saved_q2_valid_rejects_unsupported_manifest_temporal_pooling_variant(tmp_path: Path) -> None:
    archive = ArchiveAccessSentinel()
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-temporal-pooling-run"
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
                    "text_adapter_variant": "identity",
                    "classification_variant": "flat",
                    "temporal_position_variant": "none",
                    "temporal_pooling_variant": "unsupported",
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
        match=r"\Atemporal_pooling_variant must be one of: attention, attention_availability, masked_mean\Z",
    ):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-temporal-pooling-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )

    assert object.__getattribute__(archive, "accesses") == []


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


def test_run_q2_passes_scalar_mix_variant_to_local_encoder(monkeypatch, tmp_path: Path) -> None:
    config = replace(runner_config(tmp_path), text_encoder_variant="last4_scalar_mix")
    observed_variants: list[str] = []
    encoder = ScalarMixTokenEncoder()

    class RecordingEncoderFactory:
        @classmethod
        def from_local(cls, model_path: Path, *, device: torch.device, text_encoder_variant: str) -> ScalarMixTokenEncoder:
            assert model_path == config.bert_model
            assert device == torch.device("cpu")
            observed_variants.append(text_encoder_variant)
            return encoder

    monkeypatch.setattr(q2_runner, "FrozenBertEncoder", RecordingEncoderFactory)

    summary = run_q2(config, archive=runner_archive_with_inaccessible_test())

    assert summary["best_epoch"] == 0
    assert observed_variants == ["last4_scalar_mix"]


def test_run_q2_adds_five_scalar_mix_values_to_adamw_and_none_for_default(monkeypatch, tmp_path: Path) -> None:
    config = runner_config(tmp_path)
    scalar_encoder = ScalarMixTokenEncoder()
    parameter_value_counts: list[int] = []
    original_adamw = torch.optim.AdamW

    def recording_adamw(parameters, *args, **kwargs):
        parameter_values = tuple(parameters)
        parameter_value_counts.append(sum(parameter.numel() for parameter in parameter_values))
        return original_adamw(parameter_values, *args, **kwargs)

    monkeypatch.setattr(q2_runner.torch.optim, "AdamW", recording_adamw)

    run_q2(
        replace(config, output_dir=tmp_path / "scalar-mix-output", text_encoder_variant="last4_scalar_mix"),
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=scalar_encoder,
    )
    run_q2(
        replace(config, output_dir=tmp_path / "default-output", text_encoder_variant="last_hidden_state"),
        archive=runner_archive_with_inaccessible_test(),
        token_encoder=TinyTokenEncoder(),
    )

    assert sum(parameter.numel() for parameter in scalar_encoder.trainable_parameters()) == 5
    assert parameter_value_counts[0] == parameter_value_counts[1] + 5


def test_run_q2_rejects_incomplete_scalar_mix_encoder_before_training(monkeypatch, tmp_path: Path) -> None:
    class IncompleteScalarMixTokenEncoder(TinyTokenEncoder):
        def __init__(self) -> None:
            self.layer_logits = torch.nn.Parameter(torch.zeros(4))
            self.scale = torch.nn.Parameter(torch.ones(()))

        def trainable_parameters(self) -> tuple[torch.nn.Parameter, ...]:
            return (self.layer_logits, self.scale)

    def fail_if_training_starts(*args, **kwargs):
        raise AssertionError("incomplete scalar-mix encoder must be rejected before training")

    monkeypatch.setattr(q2_runner, "_train_epoch", fail_if_training_starts)

    with pytest.raises(ValueError, match="trainable_state_dict"):
        run_q2(
            replace(runner_config(tmp_path), text_encoder_variant="last4_scalar_mix"),
            archive=runner_archive_with_inaccessible_test(),
            token_encoder=IncompleteScalarMixTokenEncoder(),
        )


def test_run_q2_rejects_malformed_scalar_mix_state_before_training(monkeypatch, tmp_path: Path) -> None:
    class MalformedScalarMixTokenEncoder(ScalarMixTokenEncoder):
        def trainable_state_dict(self) -> dict[str, torch.Tensor]:
            return {"layer_logits": self.layer_logits.detach().cpu().clone()}

    def fail_if_training_starts(*args, **kwargs):
        raise AssertionError("malformed scalar-mix state must be rejected before training")

    monkeypatch.setattr(q2_runner, "_train_epoch", fail_if_training_starts)

    with pytest.raises(ValueError, match="scalar-mix trainable state"):
        run_q2(
            replace(runner_config(tmp_path), text_encoder_variant="last4_scalar_mix"),
            archive=runner_archive_with_inaccessible_test(),
            token_encoder=MalformedScalarMixTokenEncoder(),
        )


def test_check_q2_passes_scalar_mix_variant_to_local_encoder(monkeypatch, tmp_path: Path) -> None:
    config = replace(runner_config(tmp_path), text_encoder_variant="last4_scalar_mix")
    observed_variants: list[str] = []

    class RecordingEncoderFactory:
        @classmethod
        def from_local(cls, model_path: Path, *, device: torch.device, text_encoder_variant: str) -> TinyTokenEncoder:
            assert model_path == config.bert_model
            assert device == torch.device("cpu")
            observed_variants.append(text_encoder_variant)
            return TinyTokenEncoder()

    monkeypatch.setattr(q2_runner, "FrozenBertEncoder", RecordingEncoderFactory)

    summary = check_q2(config, archive=runner_archive_with_inaccessible_test())

    assert summary == {"train_count": 3, "valid_count": 3, "attachment3_count": 30}
    assert observed_variants == ["last4_scalar_mix"]


def test_run_q2_restores_best_scalar_mix_state_before_all_final_outputs(monkeypatch, tmp_path: Path) -> None:
    class RecordingScalarMixTokenEncoder(ScalarMixTokenEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.encode_scales: list[float] = []

        def encode(self, token_rows: torch.Tensor) -> torch.Tensor:
            self.encode_scales.append(float(self.scale.detach()))
            return super().encode(token_rows)

    config = replace(
        runner_config(tmp_path),
        epochs=2,
        text_encoder_variant="last4_scalar_mix",
    )
    encoder = RecordingScalarMixTokenEncoder()
    training_scales = iter((2.0, 9.0))
    evaluation_scales: list[float] = []

    def controlled_train_epoch(model, token_encoder, *args, **kwargs):
        with torch.no_grad():
            token_encoder.scale.fill_(next(training_scales))

    def controlled_evaluate(model, token_encoder, split, *args, **kwargs):
        evaluation_scales.append(float(token_encoder.scale.detach()))
        macro_f1 = 0.9 if len(evaluation_scales) == 1 else 0.8
        return (
            {"accuracy": macro_f1, "macro_f1": macro_f1, "mae": 0.0, "pearson": 0.0},
            np.zeros(split.sample_count, dtype=np.int64),
            np.zeros(split.sample_count, dtype=np.float32),
        )

    monkeypatch.setattr(q2_runner, "_train_epoch", controlled_train_epoch)
    monkeypatch.setattr(q2_runner, "_evaluate", controlled_evaluate)

    summary = run_q2(config, archive=runner_archive_with_inaccessible_test(), token_encoder=encoder)

    saved_state = torch.load(config.output_dir / "text_encoder_state.pt", weights_only=True)
    assert summary["best_epoch"] == 0
    assert evaluation_scales[:2] == [2.0, 9.0]
    assert evaluation_scales[2:] == [2.0] * 28
    assert encoder.encode_scales == [2.0] * 30
    assert torch.equal(saved_state["scale"], torch.tensor(2.0))


def test_run_q2_persists_and_restores_scalar_mix_state_without_accessing_test(tmp_path: Path) -> None:
    config = replace(runner_config(tmp_path), text_encoder_variant="last4_scalar_mix")
    encoder = ScalarMixTokenEncoder()

    summary = run_q2(config, archive=runner_archive_with_inaccessible_test(), token_encoder=encoder)

    manifest = json.loads((config.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    encoder_state = torch.load(config.output_dir / "text_encoder_state.pt", weights_only=True)
    with (config.output_dir / "validation_scenarios.csv").open(encoding="utf-8", newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    with (config.output_dir / "attachment3_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))

    assert summary["best_epoch"] == 0
    assert manifest["training"]["text_encoder_variant"] == "last4_scalar_mix"
    assert set(encoder_state) == {"layer_logits", "scale"}
    assert len(encoder.loaded_states) == 1
    assert all(torch.equal(encoder_state[name], encoder.loaded_states[0][name]) for name in encoder_state)
    assert len(scenarios) == 27
    assert len(predictions) == 30


def _write_saved_scalar_mix_run(
    run_dir: Path,
    config: Q2Config,
    *,
    text_encoder_variant: str = "last4_scalar_mix",
) -> None:
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
                    "text_encoder_variant": text_encoder_variant,
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


def test_evaluate_saved_q2_valid_strictly_restores_scalar_mix_encoder_state(monkeypatch, tmp_path: Path) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-scalar-mix-run"
    _write_saved_scalar_mix_run(run_dir, config)
    expected_state = {
        "layer_logits": torch.tensor([0.1, -0.2, 0.3, -0.4]),
        "scale": torch.tensor(1.5),
    }
    torch.save(expected_state, run_dir / "text_encoder_state.pt")
    encoder = ScalarMixTokenEncoder()
    observed_variants: list[str] = []
    observed_strict: list[bool] = []
    original_load_state_dict = MaskAwareTemporalFusion.load_state_dict

    class RecordingEncoderFactory:
        @classmethod
        def from_local(cls, model_path: Path, *, device: torch.device, text_encoder_variant: str) -> ScalarMixTokenEncoder:
            assert model_path == config.bert_model
            assert device == torch.device("cpu")
            observed_variants.append(text_encoder_variant)
            return encoder

    def recording_load_state_dict(self, *args, **kwargs):
        observed_strict.append(kwargs["strict"])
        return original_load_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(q2_runner, "FrozenBertEncoder", RecordingEncoderFactory)
    monkeypatch.setattr(MaskAwareTemporalFusion, "load_state_dict", recording_load_state_dict)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "scalar-mix-valid-report.json",
        archive=RunnerArchive(
            {
                ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                    {
                        "train": runner_split([0, 1, 2]),
                        "valid": runner_split([0, 1, 2]),
                        "test": {"not": "an accessible Q2 runtime input"},
                    }
                )
            }
        ),
    )

    assert report["sample_count"] == 3
    assert observed_variants == ["last4_scalar_mix"]
    assert observed_strict == [True]
    assert len(encoder.loaded_states) == 1
    assert all(torch.equal(encoder.loaded_states[0][name], expected_state[name]) for name in expected_state)


def test_evaluate_saved_q2_valid_defaults_historical_manifest_to_final_layer_without_state_file(
    monkeypatch, tmp_path: Path
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "historical-run"
    _write_saved_scalar_mix_run(run_dir, config)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["training"]["text_encoder_variant"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed_variants: list[str] = []

    class RecordingEncoderFactory:
        @classmethod
        def from_local(cls, model_path: Path, *, device: torch.device, text_encoder_variant: str) -> TinyTokenEncoder:
            observed_variants.append(text_encoder_variant)
            return TinyTokenEncoder()

    monkeypatch.setattr(q2_runner, "FrozenBertEncoder", RecordingEncoderFactory)

    report = evaluate_saved_q2_valid(
        run_dir,
        tmp_path / "historical-valid-report.json",
        archive=RunnerArchive(
            {
                ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                    {
                        "train": runner_split([0, 1, 2]),
                        "valid": runner_split([0, 1, 2]),
                        "test": {"not": "an accessible Q2 runtime input"},
                    }
                )
            }
        ),
    )

    assert report["sample_count"] == 3
    assert observed_variants == ["last_hidden_state"]
    assert not (run_dir / "text_encoder_state.pt").exists()


@pytest.mark.parametrize(
    ("variant", "state", "message"),
    [
        ("unsupported", None, "text_encoder_variant must be one of: last_hidden_state, last4_scalar_mix"),
        ("last4_scalar_mix", None, "text_encoder_state.pt"),
        ("last4_scalar_mix", {"layer_logits": torch.zeros(4)}, "scalar-mix trainable state"),
    ],
)
def test_evaluate_saved_q2_valid_rejects_invalid_or_unrestorable_scalar_mix_state(
    tmp_path: Path, variant: str, state: object, message: str
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "invalid-scalar-mix-run"
    _write_saved_scalar_mix_run(run_dir, config, text_encoder_variant=variant)
    if state is not None:
        torch.save(state, run_dir / "text_encoder_state.pt")

    with pytest.raises(ValueError, match=message):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-scalar-mix-report.json",
            archive=RunnerArchive(
                {
                    ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                        {
                            "train": runner_split([0, 1, 2]),
                            "valid": runner_split([0, 1, 2]),
                            "test": {"not": "an accessible Q2 runtime input"},
                        }
                    )
                }
            ),
            token_encoder=ScalarMixTokenEncoder(),
        )


@pytest.mark.parametrize(
    ("variant", "dropout", "classification_variant", "message"),
    [
        ("unsupported", 0.0, "flat", "dropout_consistency_variant must be one of: none, rdrop_alpha_1"),
        ("rdrop_alpha_1", 0.0, "flat", "rdrop_alpha_1 requires a finite dropout > 0"),
        ("rdrop_alpha_1", 0.1, "corn", "rdrop_alpha_1 requires classification_variant=flat"),
    ],
)
def test_evaluate_saved_q2_valid_rejects_invalid_dropout_semantic_before_archive_or_model(
    monkeypatch,
    tmp_path: Path,
    variant: str,
    dropout: float,
    classification_variant: str,
    message: str,
) -> None:
    config = runner_config(tmp_path)
    run_dir = tmp_path / "saved-invalid-rdrop-run"
    _write_saved_scalar_mix_run(run_dir, config, text_encoder_variant="last_hidden_state")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"].update(
        {
            "dropout_consistency_variant": variant,
            "dropout": dropout,
            "classification_variant": classification_variant,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    archive = RunnerArchive(
        {
            ALIGNED_50_MEMBER: TrainValidPayloadWithInaccessibleTest(
                {
                    "train": runner_split([0, 1, 2]),
                    "valid": runner_split([0, 1, 2]),
                    "test": {"not": "a valid evaluation input"},
                }
            )
        }
    )

    def fail_model_construction(*args, **kwargs):
        raise AssertionError("invalid semantic must be rejected before model construction")

    monkeypatch.setattr(q2_runner, "MaskAwareTemporalFusion", fail_model_construction)

    with pytest.raises(
        ValueError,
        match=rf"\A{message}\Z",
    ):
        evaluate_saved_q2_valid(
            run_dir,
            tmp_path / "invalid-rdrop-valid-report.json",
            archive=archive,
            token_encoder=TinyTokenEncoder(),
        )

    assert archive.verify_count == 0
    assert not (tmp_path / "invalid-rdrop-valid-report.json").exists()


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
