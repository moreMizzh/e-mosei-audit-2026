"""Training, validation, and artefact writing for Problem 2."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import nn

from e_mosei_audit.archive import SevenZipArchive
from e_mosei_audit.q2.config import (
    Q2Config,
    validate_classification_variant,
    validate_fusion_variant,
    validate_temporal_position_variant,
    validate_text_adapter_variant,
)
from e_mosei_audit.q2.data import AlignedSplit, Attachment3Sample, load_aligned_train_valid, load_attachment3_aligned
from e_mosei_audit.q2.missingness import (
    DroppedMasks,
    FeatureNormalizer,
    ModalityMasks,
    apply_contiguous_drop,
    apply_validation_scenario,
    fit_normalizer,
    observed_masks,
    validation_scenarios,
)
from e_mosei_audit.q2.model import FrozenBertEncoder, MaskAwareTemporalFusion, Q2Output, TensorMasks


_POLARITY_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


@dataclass(frozen=True)
class Prediction:
    """One unlabelled Attachment 3 prediction with its archive provenance."""

    sample_id: str
    source_file: str
    polarity_class: int
    sentiment_intensity: float


class TokenEncoder(Protocol):
    """The small injectable boundary between archive data and the frozen BERT feature model."""

    def encode(self, token_rows: torch.Tensor) -> torch.Tensor: ...


def compute_metrics(
    *,
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> dict[str, float | None]:
    """Compute the four task metrics from one aligned validation prediction set."""

    classes = _as_vector(true_classes, "true_classes")
    predicted = _as_vector(predicted_classes, "predicted_classes")
    scores = _as_vector(true_scores, "true_scores")
    predicted_score = _as_vector(predicted_scores, "predicted_scores")
    if not len(classes) or any(len(value) != len(classes) for value in (predicted, scores, predicted_score)):
        raise ValueError("metric inputs must be non-empty vectors with equal lengths")
    accuracy = float(np.mean(classes == predicted))
    f1_values: list[float] = []
    for label in (0, 1, 2):
        true_positive = int(np.sum((classes == label) & (predicted == label)))
        false_positive = int(np.sum((classes != label) & (predicted == label)))
        false_negative = int(np.sum((classes == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    pearson: float | None
    if np.ptp(scores) == 0 or np.ptp(predicted_score) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(scores, predicted_score)[0, 1])
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "mae": float(np.mean(np.abs(scores - predicted_score))),
        "pearson": pearson,
    }


def classification_report(
    *,
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
) -> dict[str, object]:
    """Return the fixed three-class confusion matrix and per-class validation metrics."""

    classes = _as_vector(true_classes, "true_classes")
    predicted = _as_vector(predicted_classes, "predicted_classes")
    if not len(classes) or len(predicted) != len(classes):
        raise ValueError("classification inputs must be non-empty vectors with equal lengths")
    if not np.isin(classes, (0, 1, 2)).all() or not np.isin(predicted, (0, 1, 2)).all():
        raise ValueError("classes must only contain 0, 1, or 2")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for true_class, predicted_class in zip(classes, predicted, strict=True):
        confusion[int(true_class), int(predicted_class)] += 1
    labels = ("Negative", "Neutral", "Positive")
    per_class: dict[str, dict[str, int | float]] = {}
    for label, name in enumerate(labels):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum() - true_positive)
        false_negative = int(confusion[label, :].sum() - true_positive)
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
        per_class[name] = {
            "label": label,
            "support": int(confusion[label, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        }
    return {
        "confusion_matrix": {
            "rows": "true_class",
            "columns": "predicted_class",
            "labels": list(labels),
            "counts": confusion.tolist(),
        },
        "per_class": per_class,
    }


def write_predictions(output_dir: Path, predictions: list[Prediction]) -> Path:
    """Write the final Attachment 3 prediction table with task-defined labels."""

    path = output_dir / "attachment3_predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "source_file",
                "polarity_class",
                "polarity_label",
                "sentiment_intensity",
            ),
        )
        writer.writeheader()
        for prediction in predictions:
            if prediction.polarity_class not in _POLARITY_LABELS:
                raise ValueError("polarity_class must be one of 0, 1, 2")
            if not np.isfinite(prediction.sentiment_intensity):
                raise ValueError("sentiment_intensity must be finite")
            writer.writerow(
                {
                    "sample_id": prediction.sample_id,
                    "source_file": prediction.source_file,
                    "polarity_class": prediction.polarity_class,
                    "polarity_label": _POLARITY_LABELS[prediction.polarity_class],
                    "sentiment_intensity": prediction.sentiment_intensity,
                }
            )
    return path


def run_q2(
    config: Q2Config,
    *,
    archive: Any | None = None,
    token_encoder: TokenEncoder | None = None,
) -> dict[str, int | float | None]:
    """Train on aligned Attachment 2 and make final predictions for aligned Attachment 3."""

    _validate_output_target(config.output_dir)
    active_archive = archive or SevenZipArchive(config.archive, config.seven_zip)
    active_archive.verify()
    dataset = load_aligned_train_valid(active_archive)
    attachment3 = load_attachment3_aligned(active_archive, require_complete=True)
    device = _resolve_device(config.device)
    _seed_everything(config.seed)
    active_encoder = token_encoder or FrozenBertEncoder.from_local(config.bert_model, device=device)
    train_masks = observed_masks(dataset.train.text_bert, dataset.train.audio, dataset.train.vision)
    normalizer = fit_normalizer(dataset.train.audio, dataset.train.vision, train_masks)
    valid_masks = observed_masks(dataset.valid.text_bert, dataset.valid.audio, dataset.valid.vision)
    model = MaskAwareTemporalFusion(
        hidden_size=config.hidden_size,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
        fusion_variant=config.fusion_variant,
        temporal_position_variant=config.temporal_position_variant,
        text_adapter_variant=config.text_adapter_variant,
        classification_variant=config.classification_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = _class_weights(
        dataset.train.classification_labels,
        device,
        exponent=config.class_weight_exponent,
    )
    best_epoch = -1
    best_metrics: dict[str, float | None] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    generator = np.random.default_rng(config.seed)
    for epoch in range(config.epochs):
        _train_epoch(
            model,
            active_encoder,
            optimizer,
            dataset.train,
            train_masks,
            normalizer,
            batch_size=config.batch_size,
            class_weights=weights,
            regression_loss_weight=config.regression_loss_weight,
            polarity_consistency_loss_weight=config.polarity_consistency_loss_weight,
            synthetic_missingness_enabled=config.synthetic_missingness_enabled,
            rng=generator,
            device=device,
        )
        metrics, _, _ = _evaluate(model, active_encoder, dataset.valid, valid_masks, normalizer, config.batch_size, device)
        if _is_better(metrics, best_metrics):
            best_epoch = epoch
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None or best_metrics is None:
        raise RuntimeError("no Q2 training epoch produced validation metrics")
    model.load_state_dict(best_state)
    _, valid_classes, _ = _evaluate(
        model, active_encoder, dataset.valid, valid_masks, normalizer, config.batch_size, device
    )
    valid_report = classification_report(
        true_classes=dataset.valid.classification_labels,
        predicted_classes=valid_classes,
    )
    scenario_rows = _scenario_rows(model, active_encoder, dataset.valid, valid_masks, normalizer, config.batch_size, device)
    predictions = _predict_attachment3(model, active_encoder, attachment3, normalizer, device)
    _write_run_outputs(
        config,
        model,
        normalizer,
        best_epoch=best_epoch,
        clean_metrics=best_metrics,
        valid_report=valid_report,
        scenario_rows=scenario_rows,
        predictions=predictions,
        attachment3=attachment3,
    )
    return {
        "best_epoch": best_epoch,
        "attachment3_count": len(predictions),
        **best_metrics,
    }


def check_q2(
    config: Q2Config,
    *,
    archive: Any | None = None,
    token_encoder: TokenEncoder | None = None,
) -> dict[str, int]:
    """Validate all read-only Q2 inputs without training or creating an output."""

    _validate_output_target(config.output_dir)
    active_archive = archive or SevenZipArchive(config.archive, config.seven_zip)
    active_archive.verify()
    dataset = load_aligned_train_valid(active_archive)
    attachment3 = load_attachment3_aligned(active_archive, require_complete=True)
    device = _resolve_device(config.device)
    if token_encoder is None:
        FrozenBertEncoder.from_local(config.bert_model, device=device)
    return {
        "train_count": dataset.train.sample_count,
        "valid_count": dataset.valid.sample_count,
        "attachment3_count": len(attachment3),
    }


def evaluate_saved_q2_valid(
    run_dir: Path,
    output_path: Path,
    *,
    archive: Any | None = None,
    token_encoder: TokenEncoder | None = None,
) -> dict[str, object]:
    """Evaluate one saved Q2 checkpoint on Attachment 2 valid only."""

    resolved_run_dir = run_dir.resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    model_path = resolved_run_dir / "model.pt"
    if not manifest_path.is_file() or not model_path.is_file():
        raise ValueError("run_dir must contain run_manifest.json and model.pt")
    _validate_report_target(output_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a mapping")
    training = _manifest_mapping(manifest, "training")
    normalizer_values = _manifest_mapping(manifest, "normalizer")
    device = _resolve_device(_manifest_string(training, "device"))
    active_archive = archive or SevenZipArchive(
        Path(_manifest_string(manifest, "archive")),
        Path(_manifest_string(manifest, "seven_zip")),
    )
    active_archive.verify()
    dataset = load_aligned_train_valid(active_archive)
    encoder = token_encoder or FrozenBertEncoder.from_local(
        Path(_manifest_string(manifest, "bert_model")), device=device
    )
    normalizer = FeatureNormalizer(
        audio_mean=_manifest_normalizer_array(normalizer_values, "audio_mean", 74),
        audio_std=_manifest_normalizer_array(normalizer_values, "audio_std", 74),
        vision_mean=_manifest_normalizer_array(normalizer_values, "vision_mean", 35),
        vision_std=_manifest_normalizer_array(normalizer_values, "vision_std", 35),
    )
    model = MaskAwareTemporalFusion(
        hidden_size=_manifest_positive_int(training, "hidden_size"),
        heads=_manifest_positive_int(training, "heads"),
        layers=_manifest_positive_int(training, "layers"),
        dropout=_manifest_dropout(training, "dropout"),
        fusion_variant=_manifest_fusion_variant(training),
        temporal_position_variant=_manifest_temporal_position_variant(training),
        text_adapter_variant=_manifest_text_adapter_variant(training),
        classification_variant=_manifest_classification_variant(training),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    valid_masks = observed_masks(dataset.valid.text_bert, dataset.valid.audio, dataset.valid.vision)
    metrics, predicted_classes, predicted_scores = _evaluate(
        model,
        encoder,
        dataset.valid,
        valid_masks,
        normalizer,
        _manifest_positive_int(training, "batch_size"),
        device,
    )
    report = {
        "scope": "Attachment 2 valid only; Attachment 2 test was not validated, evaluated, or used.",
        "source_run": str(resolved_run_dir),
        "sample_count": dataset.valid.sample_count,
        "metrics": metrics,
        **classification_report(
            true_classes=dataset.valid.classification_labels,
            predicted_classes=predicted_classes,
        ),
        "prediction_score_summary": {
            "mean": float(predicted_scores.mean()),
            "std": float(predicted_scores.std()),
            "min": float(predicted_scores.min()),
            "max": float(predicted_scores.max()),
        },
    }
    _write_json_new(output_path, report)
    return report


def _validate_output_target(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Q2 output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ValueError(f"Q2 output directory parent must be an existing directory: {output_dir.parent}")


def _validate_report_target(output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Q2 valid report already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise ValueError(f"Q2 valid report parent must be an existing directory: {output_path.parent}")


def _manifest_mapping(manifest: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = _manifest_value(manifest, field)
    if not isinstance(value, Mapping):
        raise ValueError(f"run manifest {field} must be a mapping")
    return value


def _manifest_string(manifest: Mapping[str, object], field: str) -> str:
    value = _manifest_value(manifest, field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run manifest {field} must be a non-empty string")
    return value


def _manifest_value(manifest: Mapping[str, object], field: str) -> object:
    if field not in manifest:
        raise ValueError(f"run manifest {field} is missing")
    return manifest[field]


def _manifest_positive_int(manifest: Mapping[str, object], field: str) -> int:
    value = _manifest_value(manifest, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"run manifest {field} must be a positive integer")
    return value


def _manifest_dropout(manifest: Mapping[str, object], field: str) -> float:
    value = _manifest_value(manifest, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"run manifest {field} must be finite")
    if not 0 <= value < 1:
        raise ValueError(f"run manifest {field} must be in [0, 1)")
    return float(value)


def _manifest_fusion_variant(training: Mapping[str, object]) -> str:
    """Treat historical saved runs as the original gated architecture."""

    if "fusion_variant" not in training:
        return "gated"
    return validate_fusion_variant(training["fusion_variant"])


def _manifest_temporal_position_variant(training: Mapping[str, object]) -> str:
    """Treat historical saved runs as having no temporal position encoding."""

    if "temporal_position_variant" not in training:
        return "none"
    return validate_temporal_position_variant(training["temporal_position_variant"])


def _manifest_text_adapter_variant(training: Mapping[str, object]) -> str:
    """Treat historical saved runs as having no text output adapter."""

    if "text_adapter_variant" not in training:
        return "identity"
    return validate_text_adapter_variant(training["text_adapter_variant"])


def _manifest_classification_variant(training: Mapping[str, object]) -> str:
    """Treat historical saved runs as using the original flat classifier."""

    if "classification_variant" not in training:
        return "flat"
    return validate_classification_variant(training["classification_variant"])


def _manifest_normalizer_array(manifest: Mapping[str, object], field: str, width: int) -> np.ndarray:
    if field not in manifest:
        raise ValueError(f"run manifest normalizer missing {field}")
    value = manifest[field]
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"run manifest normalizer {field} must be numeric") from error
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"run manifest normalizer {field} must be a finite vector of length {width}")
    return array


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Q2 config requests CUDA but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(labels: np.ndarray, device: torch.device, *, exponent: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("training split must contain each polarity class")
    if exponent == 1.0:
        weights = counts.sum() / (3.0 * counts)
        return torch.as_tensor(weights, device=device)
    stable_counts = counts.astype(np.float64)
    relative = (stable_counts.min() / stable_counts) ** exponent
    weights = relative * stable_counts.sum() / np.sum(stable_counts * relative)
    return torch.as_tensor(weights.astype(np.float32), device=device)


def _train_epoch(
    model: MaskAwareTemporalFusion,
    encoder: TokenEncoder,
    optimizer: torch.optim.Optimizer,
    split: AlignedSplit,
    masks: ModalityMasks,
    normalizer: FeatureNormalizer,
    *,
    batch_size: int,
    class_weights: torch.Tensor,
    regression_loss_weight: float,
    polarity_consistency_loss_weight: float,
    synthetic_missingness_enabled: bool,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    model.train()
    for indexes in _batch_indexes(split.sample_count, batch_size, rng):
        batch_masks = _slice_masks(masks, indexes)
        if synthetic_missingness_enabled:
            count = int(rng.integers(1, 3))
            chosen = tuple(rng.choice(np.asarray(("text", "audio", "vision")), size=count, replace=False).tolist())
            dropped = apply_contiguous_drop(batch_masks, rng=rng, modalities=chosen)
        else:
            dropped = DroppedMasks(masks=batch_masks, drops=())
        output, labels, scores = _forward_split(model, encoder, split, indexes, dropped, normalizer, device)
        loss = _joint_loss(
            output,
            labels,
            scores,
            class_weights,
            regression_loss_weight=regression_loss_weight,
            polarity_consistency_loss_weight=polarity_consistency_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _joint_loss(
    output: Q2Output,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    regression_loss_weight: float,
    polarity_consistency_loss_weight: float,
) -> torch.Tensor:
    classification = _classification_loss(output, labels, class_weights)
    regression = nn.functional.smooth_l1_loss(output.score, scores)
    probabilities = torch.softmax(output.logits, dim=1)
    expected_polarity = probabilities @ output.logits.new_tensor([-1.0, 0.0, 1.0])
    consistency = nn.functional.smooth_l1_loss(output.score / 3.0, expected_polarity)
    return classification + regression_loss_weight * regression + polarity_consistency_loss_weight * consistency


def _classification_loss(output: Q2Output, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Use flat cross-entropy or CORN's two conditional binary objectives."""

    if output.ordinal_logits is None:
        return nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
    ordinal_logits = output.ordinal_logits
    sample_weights = class_weights[labels]
    first_terms = nn.functional.binary_cross_entropy_with_logits(
        ordinal_logits[:, 0],
        (labels > 0).to(ordinal_logits.dtype),
        reduction="none",
    )
    first = (first_terms * sample_weights).sum() / sample_weights.sum()
    active = labels > 0
    terms = [first]
    if bool(active.any()):
        second_weights = sample_weights[active]
        second_terms = nn.functional.binary_cross_entropy_with_logits(
            ordinal_logits[active, 1],
            (labels[active] > 1).to(ordinal_logits.dtype),
            reduction="none",
        )
        terms.append(
            (second_terms * second_weights).sum() / second_weights.sum()
        )
    return torch.stack(terms).mean()


def _evaluate(
    model: MaskAwareTemporalFusion,
    encoder: TokenEncoder,
    split: AlignedSplit,
    masks: ModalityMasks,
    normalizer: FeatureNormalizer,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    model.eval()
    classes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for indexes in _batch_indexes(split.sample_count, batch_size, None):
            output, _, _ = _forward_split(
                model,
                encoder,
                split,
                indexes,
                DroppedMasks(masks=_slice_masks(masks, indexes), drops=()),
                normalizer,
                device,
            )
            classes.append(output.logits.argmax(dim=1).cpu().numpy())
            scores.append(output.score.cpu().numpy())
    predicted_classes = np.concatenate(classes)
    predicted_scores = np.concatenate(scores)
    return (
        compute_metrics(
            true_classes=split.classification_labels,
            predicted_classes=predicted_classes,
            true_scores=split.regression_labels,
            predicted_scores=predicted_scores,
        ),
        predicted_classes,
        predicted_scores,
    )


def _forward_split(
    model: MaskAwareTemporalFusion,
    encoder: TokenEncoder,
    split: AlignedSplit,
    indexes: np.ndarray,
    dropped: DroppedMasks,
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    tokens = torch.as_tensor(split.text_bert[indexes], dtype=torch.int64, device=device)
    text = encoder.encode(tokens)
    audio = torch.as_tensor(_normalise_audio(split.audio[indexes], normalizer), device=device)
    vision = torch.as_tensor(_normalise_vision(split.vision[indexes], normalizer), device=device)
    output = model(text=text, audio=audio, vision=vision, masks=_tensor_masks(dropped.masks, device))
    labels = torch.as_tensor(split.classification_labels[indexes], dtype=torch.int64, device=device)
    scores = torch.as_tensor(split.regression_labels[indexes], dtype=torch.float32, device=device)
    return output, labels, scores


def _batch_indexes(sample_count: int, batch_size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    indexes = np.arange(sample_count) if rng is None else rng.permutation(sample_count)
    return [indexes[start : start + batch_size] for start in range(0, sample_count, batch_size)]


def _slice_masks(masks: ModalityMasks, indexes: np.ndarray) -> ModalityMasks:
    return ModalityMasks(
        text=masks.text[indexes],
        audio=masks.audio[indexes],
        vision=masks.vision[indexes],
        temporal=masks.temporal[indexes],
    )


def _tensor_masks(masks: ModalityMasks, device: torch.device) -> TensorMasks:
    return TensorMasks(
        text=torch.as_tensor(masks.text, dtype=torch.bool, device=device),
        audio=torch.as_tensor(masks.audio, dtype=torch.bool, device=device),
        vision=torch.as_tensor(masks.vision, dtype=torch.bool, device=device),
        temporal=torch.as_tensor(masks.temporal, dtype=torch.bool, device=device),
    )


def _normalise_audio(values: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
    return ((values - normalizer.audio_mean) / normalizer.audio_std).astype(np.float32, copy=False)


def _normalise_vision(values: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
    return ((values - normalizer.vision_mean) / normalizer.vision_std).astype(np.float32, copy=False)


def _is_better(candidate: dict[str, float | None], incumbent: dict[str, float | None] | None) -> bool:
    if incumbent is None:
        return True
    candidate_f1 = candidate["macro_f1"]
    incumbent_f1 = incumbent["macro_f1"]
    assert isinstance(candidate_f1, float) and isinstance(incumbent_f1, float)
    if candidate_f1 != incumbent_f1:
        return candidate_f1 > incumbent_f1
    candidate_mae = candidate["mae"]
    incumbent_mae = incumbent["mae"]
    assert isinstance(candidate_mae, float) and isinstance(incumbent_mae, float)
    return candidate_mae < incumbent_mae


def _scenario_rows(
    model: MaskAwareTemporalFusion,
    encoder: TokenEncoder,
    split: AlignedSplit,
    masks: ModalityMasks,
    normalizer: FeatureNormalizer,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in validation_scenarios():
        dropped = apply_validation_scenario(masks, scenario)
        metrics, _, _ = _evaluate(model, encoder, split, dropped.masks, normalizer, batch_size, device)
        target_available = int(np.count_nonzero(getattr(masks, scenario.modality)))
        dropped_available = int(
            np.count_nonzero(getattr(masks, scenario.modality) & ~getattr(dropped.masks, scenario.modality))
        )
        rows.append(
            {
                "modality": scenario.modality,
                "position": scenario.position,
                "fraction": scenario.fraction,
                "target_available_positions": target_available,
                "dropped_available_positions": dropped_available,
                "actual_available_fraction": dropped_available / target_available if target_available else 0.0,
                **metrics,
            }
        )
    return rows


def _predict_attachment3(
    model: MaskAwareTemporalFusion,
    encoder: TokenEncoder,
    samples: list[Attachment3Sample],
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> list[Prediction]:
    model.eval()
    predictions: list[Prediction] = []
    with torch.no_grad():
        for sample in samples:
            masks = observed_masks(sample.text_bert, sample.audio, sample.vision)
            tokens = torch.as_tensor(sample.text_bert, dtype=torch.int64, device=device)
            output = model(
                text=encoder.encode(tokens),
                audio=torch.as_tensor(_normalise_audio(sample.audio, normalizer), device=device),
                vision=torch.as_tensor(_normalise_vision(sample.vision, normalizer), device=device),
                masks=_tensor_masks(masks, device),
            )
            predictions.append(
                Prediction(
                    sample_id=sample.sample_id,
                    source_file=sample.source_file,
                    polarity_class=int(output.logits.argmax(dim=1).item()),
                    sentiment_intensity=float(output.score.item()),
                )
            )
    return predictions


def _write_run_outputs(
    config: Q2Config,
    model: MaskAwareTemporalFusion,
    normalizer: FeatureNormalizer,
    *,
    best_epoch: int,
    clean_metrics: dict[str, float | None],
    valid_report: dict[str, object],
    scenario_rows: list[dict[str, object]],
    predictions: list[Prediction],
    attachment3: list[Attachment3Sample],
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}-", dir=config.output_dir.parent))
    try:
        write_predictions(staging, predictions)
        _write_csv(staging / "validation_scenarios.csv", scenario_rows)
        _write_csv(staging / "attachment3_missingness.csv", _missingness_rows(attachment3))
        _write_json(staging / "metrics.json", {"clean": clean_metrics})
        _write_json(staging / "valid_classification_report.json", valid_report)
        manifest = {
            "archive": str(config.archive),
            "seven_zip": str(config.seven_zip),
            "bert_model": str(config.bert_model),
            "seed": config.seed,
            "best_epoch": best_epoch,
            "training": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "hidden_size": config.hidden_size,
                "heads": config.heads,
                "layers": config.layers,
                "dropout": config.dropout,
                "regression_loss_weight": config.regression_loss_weight,
                "polarity_consistency_loss_weight": config.polarity_consistency_loss_weight,
                "class_weight_exponent": config.class_weight_exponent,
                "fusion_variant": config.fusion_variant,
                "text_adapter_variant": config.text_adapter_variant,
                "classification_variant": config.classification_variant,
                "temporal_position_variant": config.temporal_position_variant,
                "device": config.device,
                "synthetic_missingness": {
                    "enabled": config.synthetic_missingness_enabled,
                    "modalities_per_sample": "1 or 2",
                    "fraction_range": [0.1, 0.5],
                },
            },
            "normalizer": normalizer.as_dict(),
            "attachment3_count": len(predictions),
        }
        if config.fusion_variant == "pooled_lmf_r4":
            manifest["architecture"] = {"pooled_lmf_rank": 4}
        _write_json(staging / "run_manifest.json", manifest)
        torch.save(model.state_dict(), staging / "model.pt")
        (staging / "audit_report.md").write_text(
            _render_audit_report(best_epoch, clean_metrics, scenario_rows), encoding="utf-8"
        )
        _publish_staging(staging, config.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_staging(staging: Path, output_dir: Path) -> None:
    """Atomically publish only if no process has claimed the output name."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Q2 output directory already exists: {output_dir}")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except AttributeError as error:  # pragma: no cover - requires a non-Linux runtime.
        raise RuntimeError("Q2 output publication requires Linux renameat2(RENAME_NOREPLACE)") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output_dir), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Q2 output directory already exists: {output_dir}")
    raise OSError(error_number, os.strerror(error_number), output_dir)


def _missingness_rows(samples: list[Attachment3Sample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        masks = observed_masks(sample.text_bert, sample.audio, sample.vision)
        for modality in ("text", "audio", "vision"):
            unavailable = ~getattr(masks, modality)[0]
            leading = _leading_count(unavailable)
            trailing = _leading_count(unavailable[::-1]) if leading < len(unavailable) else 0
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_file": sample.source_file,
                    "modality": modality,
                    "leading_unavailable": leading,
                    "trailing_unavailable": trailing,
                    "interior_unavailable": int(unavailable.sum()) - leading - trailing,
                    "longest_unavailable_run": _longest_run(unavailable),
                }
            )
    return rows


def _leading_count(values: np.ndarray) -> int:
    false_positions = np.flatnonzero(~values)
    return int(false_positions[0]) if len(false_positions) else len(values)


def _longest_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _render_audit_report(
    best_epoch: int, clean_metrics: dict[str, float | None], scenario_rows: list[dict[str, object]]
) -> str:
    lines = [
        "# 问题 2 运行报告",
        "",
        "验证指标仅来自附件 2 valid 划分。附件 3 无标签，仅生成最终推理结果，未参与模型选择或指标统计。",
        "",
        "## 清洁验证",
        "",
        f"- 最佳 epoch: {best_epoch}",
        f"- Accuracy: {_format_metric(clean_metrics['accuracy'])}",
        f"- macro-F1: {_format_metric(clean_metrics['macro_f1'])}",
        f"- MAE: {_format_metric(clean_metrics['mae'])}",
        f"- Pearson: {_format_metric(clean_metrics['pearson'])}",
        "",
        "## 缺失影响汇总",
        "",
        "下表是相对清洁验证的平均变化；Accuracy、macro-F1、Pearson 的负值表示下降，MAE 的正值表示误差增大。",
        "",
        "| 维度 | 水平 | Accuracy 变化 | macro-F1 变化 | MAE 变化 | Pearson 变化 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dimension in ("modality", "position", "fraction"):
        grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
        for row in scenario_rows:
            grouped[row[dimension]].append(row)
        for level, rows in grouped.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        dimension,
                        str(level),
                        _format_delta(rows, clean_metrics, "accuracy"),
                        _format_delta(rows, clean_metrics, "macro_f1"),
                        _format_delta(rows, clean_metrics, "mae"),
                        _format_delta(rows, clean_metrics, "pearson"),
                    )
                )
                + " |"
            )
    worst_f1 = min(scenario_rows, key=lambda row: float(row["macro_f1"]))
    worst_mae = max(scenario_rows, key=lambda row: float(row["mae"]))
    lines.extend(
        (
            "",
            "## 最不利场景",
            "",
            "- 最不利 macro-F1 场景: "
            f"{worst_f1['modality']} / {worst_f1['position']} / {worst_f1['fraction']}，"
            f"macro-F1={_format_metric(worst_f1['macro_f1'])}。",
            "- 最大 MAE 场景: "
            f"{worst_mae['modality']} / {worst_mae['position']} / {worst_mae['fraction']}，"
            f"MAE={_format_metric(worst_mae['mae'])}。",
            "",
            "全零连续段只作为题面定义的模态不可用证据；没有独立标签时，不把附件 3 的预测解释为性能指标。",
            "",
        )
    )
    return "\n".join(lines)


def _format_delta(rows: list[dict[str, object]], clean: dict[str, float | None], key: str) -> str:
    baseline = clean[key]
    values = [row[key] for row in rows if row[key] is not None]
    if baseline is None or not values:
        return "N/A"
    return f"{float(np.mean([float(value) - float(baseline) for value in values])):+.4f}"


def _format_metric(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite numeric vector")
    return vector
