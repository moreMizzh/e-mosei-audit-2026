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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
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
    assert config.dropout_consistency_variant == "none"
    assert config.class_weight_exponent == 1.0
    assert config.synthetic_missingness_enabled is True
    assert config.fusion_variant == "gated"
    assert config.text_adapter_variant == "identity"
    assert config.classification_variant == "flat"
    assert config.temporal_position_variant == "none"
    assert config.temporal_pooling_variant == "attention"
    assert config.text_encoder_variant == "last_hidden_state"
    assert config.device == "cpu"


def test_load_q2_config_requires_classification_loss_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_q2_config(config_path)

    assert error.value.args == ("missing required training field: classification_loss_variant",)


def test_load_q2_config_requires_temporal_position_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_q2_config(config_path)

    assert error.value.args == ("missing required training field: temporal_position_variant",)


def test_load_q2_config_requires_temporal_pooling_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_q2_config(config_path)

    assert error.value.args == ("missing required training field: temporal_pooling_variant",)


def test_load_q2_config_requires_text_encoder_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_q2_config(config_path)

    assert error.value.args == ("missing required training field: text_encoder_variant",)


def test_load_q2_config_requires_dropout_consistency_variant(tmp_path: Path) -> None:
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
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_q2_config(config_path)

    assert error.value.args == ("missing required training field: dropout_consistency_variant",)


@pytest.mark.parametrize(
    "variant",
    [
        "gated",
        "mag_lite",
        "mult_lite",
        "late_expert_shared",
        "text_anchor_residual",
        "pairwise_hadamard_residual",
        "pooled_lmf_r4",
    ],
)
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "{variant}"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).fusion_variant == variant


@pytest.mark.parametrize("variant", ["identity", "houlsby_output_b32"])
def test_load_q2_config_parses_supported_text_adapter_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "{variant}"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).text_adapter_variant == variant


@pytest.mark.parametrize("variant", ["flat", "corn"])
def test_load_q2_config_parses_supported_classification_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "{variant}"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).classification_variant == variant


@pytest.mark.parametrize("variant", ["hard_ce", "weighted_label_smoothing_005"])
def test_load_q2_config_parses_supported_classification_loss_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "{variant}"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).classification_loss_variant == variant


def test_load_q2_config_rejects_unsupported_classification_loss_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "unsupported"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Aclassification_loss_variant must be one of: hard_ce, weighted_label_smoothing_005\Z",
    ):
        load_q2_config(config_path)


def test_load_q2_config_rejects_smoothing_for_corn_classification(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "corn"
classification_loss_variant = "weighted_label_smoothing_005"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Aweighted_label_smoothing_005 requires classification_variant=flat\Z",
    ):
        load_q2_config(config_path)


def test_load_q2_config_rejects_smoothing_with_rdrop(tmp_path: Path) -> None:
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
dropout = 0.1
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
dropout_consistency_variant = "rdrop_alpha_1"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "weighted_label_smoothing_005"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Aweighted_label_smoothing_005 cannot be combined with rdrop_alpha_1\Z",
    ):
        load_q2_config(config_path)


@pytest.mark.parametrize(("variant", "dropout"), [("none", "0.0"), ("rdrop_alpha_1", "0.1")])
def test_load_q2_config_parses_supported_dropout_consistency_variant(
    tmp_path: Path, variant: str, dropout: str
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
dropout = {dropout}
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
dropout_consistency_variant = "{variant}"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).dropout_consistency_variant == variant


def test_load_q2_config_rejects_rdrop_without_dropout(tmp_path: Path) -> None:
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
dropout_consistency_variant = "rdrop_alpha_1"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\Ardrop_alpha_1 requires a finite dropout > 0\Z"):
        load_q2_config(config_path)


@pytest.mark.parametrize("value", ['"unsupported"', "1"])
def test_load_q2_config_rejects_invalid_dropout_consistency_variant(tmp_path: Path, value: str) -> None:
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
dropout_consistency_variant = {value}
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\Adropout_consistency_variant must be one of: none, rdrop_alpha_1\Z"):
        load_q2_config(config_path)


def test_load_q2_config_rejects_rdrop_for_corn_classification(tmp_path: Path) -> None:
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
dropout_consistency_variant = "rdrop_alpha_1"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "corn"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\Ardrop_alpha_1 requires classification_variant=flat\Z"):
        load_q2_config(config_path)


@pytest.mark.parametrize("variant", ["none", "sinusoidal"])
def test_load_q2_config_parses_supported_temporal_position_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "{variant}"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).temporal_position_variant == variant


@pytest.mark.parametrize(
    "variant",
    ["attention", "attention_availability", "attention_statistics_residual", "masked_mean"],
)
def test_load_q2_config_parses_supported_temporal_pooling_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "{variant}"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).temporal_pooling_variant == variant


@pytest.mark.parametrize("variant", ["last_hidden_state", "last4_scalar_mix"])
def test_load_q2_config_parses_supported_text_encoder_variant(tmp_path: Path, variant: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "{variant}"
device = "cpu"
''',
        encoding="utf-8",
    )

    assert load_q2_config(config_path).text_encoder_variant == variant


@pytest.mark.parametrize("value", ['"unsupported"', "0"])
def test_load_q2_config_rejects_unsupported_or_nonstring_text_encoder_variant(tmp_path: Path, value: str) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = {value}
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Atext_encoder_variant must be one of: last_hidden_state, last4_scalar_mix\Z",
    ):
        load_q2_config(config_path)


def test_load_q2_config_rejects_unsupported_temporal_position_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "unsupported"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\Atemporal_position_variant must be one of: none, sinusoidal\Z",
    ):
        load_q2_config(config_path)


def test_load_q2_config_rejects_unsupported_temporal_pooling_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "unsupported"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"\Atemporal_pooling_variant must be one of: attention, attention_availability, "
            r"attention_statistics_residual, masked_mean\Z"
        ),
    ):
        load_q2_config(config_path)


def test_load_q2_config_rejects_unsupported_classification_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "unsupported"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\Aclassification_variant must be one of: flat, corn\Z"):
        load_q2_config(config_path)


def test_load_q2_config_rejects_unsupported_text_adapter_variant(tmp_path: Path) -> None:
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "unsupported"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="text_adapter_variant must be one of: identity, houlsby_output_b32",
    ):
        load_q2_config(config_path)


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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "unsupported"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"\Afusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared, "
            r"text_anchor_residual, pairwise_hadamard_residual, pooled_lmf_r4\Z"
        ),
    ):
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = {value}
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
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
dropout_consistency_variant = "none"
class_weight_exponent = {value if field == "class_weight_exponent" else "1.0"}
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
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


def _write_temporal_context_config(
    tmp_path: Path,
    *,
    temporal_context_variant: str | None = '"none"',
    fusion_variant: str = "gated",
) -> Path:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"zip")
    seven_zip = tmp_path / "7za"
    seven_zip.write_bytes(b"tool")
    seven_zip.chmod(0o755)
    bert_model = tmp_path / "bert-base-uncased"
    bert_model.mkdir()
    context_line = (
        "" if temporal_context_variant is None else f"temporal_context_variant = {temporal_context_variant}\n"
    )
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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = true
fusion_variant = "{fusion_variant}"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
{context_line}temporal_residual_variant = "none"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cpu"
''',
        encoding="utf-8",
    )
    return config_path


def _write_temporal_residual_config(
    tmp_path: Path,
    *,
    temporal_residual_variant: str | None = '"none"',
    fusion_variant: str = "gated",
) -> Path:
    config_path = _write_temporal_context_config(tmp_path, fusion_variant=fusion_variant)
    residual_line = (
        "" if temporal_residual_variant is None else f"temporal_residual_variant = {temporal_residual_variant}\n"
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'temporal_residual_variant = "none"\ntemporal_pooling_variant = "attention"\n',
            f'{residual_line}temporal_pooling_variant = "attention"\n',
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_q2_config_requires_temporal_context_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\Amissing required training field: temporal_context_variant\Z"):
        load_q2_config(_write_temporal_context_config(tmp_path, temporal_context_variant=None))


@pytest.mark.parametrize("variant", ["none", "availability_embedding"])
def test_load_q2_config_parses_supported_temporal_context_variant(tmp_path: Path, variant: str) -> None:
    config = load_q2_config(_write_temporal_context_config(tmp_path, temporal_context_variant=f'"{variant}"'))

    assert config.temporal_context_variant == variant


@pytest.mark.parametrize("variant", ['"unsupported"', "0"])
def test_load_q2_config_rejects_invalid_temporal_context_variant(tmp_path: Path, variant: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"\Atemporal_context_variant must be one of: none, availability_embedding\Z",
    ):
        load_q2_config(_write_temporal_context_config(tmp_path, temporal_context_variant=variant))


def test_load_q2_config_rejects_availability_context_for_late_expert_fusion(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"\Aavailability_embedding temporal context is unsupported with late_expert_shared fusion\Z",
    ):
        load_q2_config(
            _write_temporal_context_config(
                tmp_path,
                temporal_context_variant='"availability_embedding"',
                fusion_variant="late_expert_shared",
            )
        )


def test_load_q2_config_requires_temporal_residual_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\Amissing required training field: temporal_residual_variant\Z"):
        load_q2_config(_write_temporal_residual_config(tmp_path, temporal_residual_variant=None))


@pytest.mark.parametrize("variant", ["none", "depthwise_conv3"])
def test_load_q2_config_parses_supported_temporal_residual_variant(tmp_path: Path, variant: str) -> None:
    config = load_q2_config(_write_temporal_residual_config(tmp_path, temporal_residual_variant=f'"{variant}"'))

    assert config.temporal_residual_variant == variant


@pytest.mark.parametrize("variant", ['"unsupported"', "0"])
def test_load_q2_config_rejects_invalid_temporal_residual_variant(tmp_path: Path, variant: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"\Atemporal_residual_variant must be one of: none, depthwise_conv3\Z",
    ):
        load_q2_config(_write_temporal_residual_config(tmp_path, temporal_residual_variant=variant))


def test_load_q2_config_rejects_depthwise_residual_for_late_expert_fusion(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"\Adepthwise_conv3 temporal residual is unsupported with late_expert_shared fusion\Z",
    ):
        load_q2_config(
            _write_temporal_residual_config(
                tmp_path,
                temporal_residual_variant='"depthwise_conv3"',
                fusion_variant="late_expert_shared",
            )
        )
