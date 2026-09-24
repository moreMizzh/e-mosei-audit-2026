"""Explicit local configuration for Problem 2 training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tomllib


_PATH_FIELDS = ("archive", "seven_zip", "bert_model", "output_dir")
FUSION_VARIANTS = ("gated", "mag_lite", "mult_lite", "late_expert_shared", "text_anchor_residual")
TEXT_ADAPTER_VARIANTS = ("identity", "houlsby_output_b32")
CLASSIFICATION_VARIANTS = ("flat", "corn")
_TRAINING_FIELDS = (
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "hidden_size",
    "heads",
    "layers",
    "dropout",
    "regression_loss_weight",
    "polarity_consistency_loss_weight",
    "class_weight_exponent",
    "synthetic_missingness_enabled",
    "fusion_variant",
    "text_adapter_variant",
    "classification_variant",
    "device",
)


@dataclass(frozen=True)
class Q2Config:
    archive: Path
    seven_zip: Path
    bert_model: Path
    output_dir: Path
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    heads: int
    layers: int
    dropout: float
    regression_loss_weight: float
    polarity_consistency_loss_weight: float
    class_weight_exponent: float
    synthetic_missingness_enabled: bool
    fusion_variant: str
    text_adapter_variant: str
    classification_variant: str
    device: str


def load_q2_config(path: Path) -> Q2Config:
    """Load an explicit Q2 TOML file without creating output or downloading assets."""

    config_path = path.resolve()
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    paths = _required_table(document, "paths")
    training = _required_table(document, "training")
    _require_exact_fields(paths, _PATH_FIELDS, "paths")
    _require_exact_fields(training, _TRAINING_FIELDS, "training")
    resolved = {name: _resolve_path(paths[name], name, config_path.parent) for name in _PATH_FIELDS}
    if resolved["archive"].suffix.lower() != ".zip":
        raise ValueError("archive must name the final .zip volume")
    _require_file(resolved["archive"], "archive")
    _require_executable(resolved["seven_zip"], "seven_zip")
    _require_directory(resolved["bert_model"], "bert_model")
    if resolved["output_dir"].exists() or resolved["output_dir"].is_symlink():
        raise FileExistsError(f"Q2 output directory already exists: {resolved['output_dir']}")
    values = {name: training[name] for name in _TRAINING_FIELDS}
    _validate_training(values)
    return Q2Config(
        archive=resolved["archive"],
        seven_zip=resolved["seven_zip"],
        bert_model=resolved["bert_model"],
        output_dir=resolved["output_dir"],
        seed=values["seed"],
        epochs=values["epochs"],
        batch_size=values["batch_size"],
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        hidden_size=values["hidden_size"],
        heads=values["heads"],
        layers=values["layers"],
        dropout=float(values["dropout"]),
        regression_loss_weight=float(values["regression_loss_weight"]),
        polarity_consistency_loss_weight=float(values["polarity_consistency_loss_weight"]),
        class_weight_exponent=float(values["class_weight_exponent"]),
        synthetic_missingness_enabled=values["synthetic_missingness_enabled"],
        fusion_variant=validate_fusion_variant(values["fusion_variant"]),
        text_adapter_variant=validate_text_adapter_variant(values["text_adapter_variant"]),
        classification_variant=validate_classification_variant(values["classification_variant"]),
        device=values["device"],
    )


def _required_table(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    table = document.get(name)
    if not isinstance(table, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return table


def _require_exact_fields(table: Mapping[str, object], fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in table]
    if missing:
        raise ValueError(f"missing required {name} field: {missing[0]}")
    unexpected = sorted(set(table) - set(fields))
    if unexpected:
        raise ValueError(f"unexpected {name} field: {unexpected[0]}")


def _resolve_path(value: object, field: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    entry = Path(value)
    return (entry if entry.is_absolute() else base_dir / entry).resolve()


def _require_file(path: Path, field: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field} must be an existing regular file: {path}")


def _require_executable(path: Path, field: str) -> None:
    _require_file(path, field)
    if not os.access(path, os.X_OK):
        raise ValueError(f"{field} must be executable: {path}")


def _require_directory(path: Path, field: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field} must be an existing directory: {path}")


def _validate_training(values: Mapping[str, object]) -> None:
    for name in ("seed", "epochs", "batch_size", "hidden_size", "heads", "layers"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'non-negative' if name == 'seed' else 'positive'} integer")
    for name in ("learning_rate", "weight_decay", "dropout"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if values["learning_rate"] <= 0 or values["weight_decay"] < 0 or not 0 <= values["dropout"] < 1:
        raise ValueError("learning_rate, weight_decay, or dropout is outside its valid range")
    regression_loss_weight = values["regression_loss_weight"]
    if (
        isinstance(regression_loss_weight, bool)
        or not isinstance(regression_loss_weight, (int, float))
        or not math.isfinite(float(regression_loss_weight))
        or regression_loss_weight < 0
    ):
        raise ValueError("regression_loss_weight is outside its valid range")
    polarity_consistency_loss_weight = values["polarity_consistency_loss_weight"]
    if (
        isinstance(polarity_consistency_loss_weight, bool)
        or not isinstance(polarity_consistency_loss_weight, (int, float))
        or not math.isfinite(float(polarity_consistency_loss_weight))
        or polarity_consistency_loss_weight < 0
    ):
        raise ValueError("polarity_consistency_loss_weight is outside its valid range")
    class_weight_exponent = values["class_weight_exponent"]
    if (
        isinstance(class_weight_exponent, bool)
        or not isinstance(class_weight_exponent, (int, float))
        or not math.isfinite(float(class_weight_exponent))
        or class_weight_exponent <= 0
    ):
        raise ValueError("class_weight_exponent is outside its valid range")
    if not isinstance(values["synthetic_missingness_enabled"], bool):
        raise ValueError("synthetic_missingness_enabled must be boolean")
    validate_fusion_variant(values["fusion_variant"])
    validate_text_adapter_variant(values["text_adapter_variant"])
    validate_classification_variant(values["classification_variant"])
    if values["hidden_size"] % values["heads"]:
        raise ValueError("hidden_size must be divisible by heads")
    if not isinstance(values["device"], str) or not values["device"]:
        raise ValueError("device must be a non-empty string")


def validate_fusion_variant(value: object) -> str:
    """Require one of the persisted Q2 fusion architecture names."""

    if not isinstance(value, str) or value not in FUSION_VARIANTS:
        raise ValueError("fusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared")
    return value


def validate_text_adapter_variant(value: object) -> str:
    """Require one of the persisted Q2 text output adapter names."""

    if not isinstance(value, str) or value not in TEXT_ADAPTER_VARIANTS:
        raise ValueError("text_adapter_variant must be one of: identity, houlsby_output_b32")
    return value


def validate_classification_variant(value: object) -> str:
    """Require one of the persisted Q2 classification parameterizations."""

    if not isinstance(value, str) or value not in CLASSIFICATION_VARIANTS:
        raise ValueError("classification_variant must be one of: flat, corn")
    return value
