from __future__ import annotations

from pathlib import Path

import pytest

from e_mosei_audit.q2.config import load_q2_config


def test_load_q2_config_resolves_relative_paths_and_training_values(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        """[paths]
archive = "data.zip"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.01
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
device = "cpu"
""",
        encoding="utf-8",
    )

    config = load_q2_config(config_path)

    assert config.archive == archive
    assert config.bert_model == bert_model
    assert config.output_dir == tmp_path / "artifacts" / "q2"
    assert config.epochs == 3
    assert config.regression_loss_weight == 0.5
    assert config.polarity_consistency_loss_weight == 0.0
    assert config.class_weight_exponent == 1.0
    assert config.synthetic_missingness_enabled is True
    assert config.fusion_variant == "gated"
    assert config.device == "cpu"


@pytest.mark.parametrize("variant", ["gated", "mag_lite", "mult_lite"])
def test_load_q2_config_parses_supported_fusion_variant(tmp_path: Path, variant: str) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        f'''[paths]
archive = "data.zip"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.01
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "{variant}"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).fusion_variant == variant


def test_load_q2_config_rejects_unsupported_fusion_variant(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        '''[paths]
archive = "data.zip"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.01
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "unsupported"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fusion_variant must be one of: gated, mag_lite, mult_lite"):
        load_q2_config(config_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("false", False), ("1", None), ('"false"', None)],
)
def test_load_q2_config_parses_synthetic_missingness_enabled(
    tmp_path: Path, value: str, expected: bool | None
) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        f'''[paths]
archive = "data.zip"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.01
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = {value}
fusion_variant = "gated"
device = "cpu"
''',
        encoding="utf-8",
    )

    if expected is None:
        with pytest.raises(ValueError, match="synthetic_missingness_enabled must be boolean"):
            load_q2_config(config_path)
    else:
        assert load_q2_config(config_path).synthetic_missingness_enabled is expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("learning_rate", "nan", "learning_rate must be finite"),
        ("weight_decay", "nan", "weight_decay must be finite"),
        ("regression_loss_weight", "nan", "regression_loss_weight is outside its valid range"),
        ("regression_loss_weight", "-0.1", "regression_loss_weight is outside its valid range"),
        (
            "polarity_consistency_loss_weight",
            "nan",
            "polarity_consistency_loss_weight is outside its valid range",
        ),
        (
            "polarity_consistency_loss_weight",
            "-0.1",
            "polarity_consistency_loss_weight is outside its valid range",
        ),
        ("class_weight_exponent", "0.0", "class_weight_exponent is outside its valid range"),
        ("class_weight_exponent", "-1.0", "class_weight_exponent is outside its valid range"),
        ("class_weight_exponent", "nan", "class_weight_exponent is outside its valid range"),
    ],
)
def test_load_q2_config_rejects_nonfinite_training_values(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        f'''[paths]
archive = "data.zip"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = {value if field == "learning_rate" else "0.001"}
weight_decay = {value if field == "weight_decay" else "0.01"}
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = {value if field == "regression_loss_weight" else "0.5"}
polarity_consistency_loss_weight = {value if field == "polarity_consistency_loss_weight" else "0.0"}
class_weight_exponent = {value if field == "class_weight_exponent" else "1.0"}
synthetic_missingness_enabled = true
fusion_variant = "gated"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_q2_config(config_path)


def test_load_q2_config_requires_a_new_output_and_executable_last_zip_volume(tmp_path: Path) -> None:
    archive = tmp_path / "data.001"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    output_dir = tmp_path / "artifacts" / "q2"
    output_dir.mkdir(parents=True)
    config_path = tmp_path / "q2.toml"
    config_path.write_text(
        '''[paths]
archive = "data.001"
seven_zip = "7za"
bert_model = "bert-base-uncased"
output_dir = "artifacts/q2"

[training]
seed = 7
epochs = 3
batch_size = 2
learning_rate = 0.001
weight_decay = 0.01
hidden_size = 16
heads = 4
layers = 1
dropout = 0.0
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="archive must name the final .zip volume"):
        load_q2_config(config_path)

    archive.rename(tmp_path / "data.zip")
    config_path.write_text(config_path.read_text(encoding="utf-8").replace("data.001", "data.zip"), encoding="utf-8")
    with pytest.raises(ValueError, match="seven_zip must be executable"):
        load_q2_config(config_path)

    seven_zip.chmod(0o755)
    with pytest.raises(FileExistsError, match="output directory already exists"):
        load_q2_config(config_path)
