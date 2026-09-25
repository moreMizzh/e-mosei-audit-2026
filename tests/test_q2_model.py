from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from e_mosei_audit.q2.model import (
    FrozenBertEncoder,
    MaskAwareTemporalFusion,
    TensorMasks,
    _POOLED_LMF_RANK,
    _pairwise_hadamard_residual,
    _pooled_lmf_residual,
    _sinusoidal_position_encoding,
)


class FakeBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.ones(1))
        self.calls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool] | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        self.calls = (input_ids, attention_mask, token_type_ids, torch.is_grad_enabled(), output_hidden_states)
        hidden_states = tuple(
            torch.full((input_ids.shape[0], input_ids.shape[1], 768), float(layer), device=input_ids.device)
            for layer in range(1, 5)
        )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states if output_hidden_states else None,
        )


class RecordingAttention(nn.Module):
    """Delegate to real attention while recording the mask passed to it."""

    def __init__(self, delegate: nn.MultiheadAttention) -> None:
        super().__init__()
        self.delegate = delegate
        self.key_padding_masks: list[torch.Tensor] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key_padding_mask is None:
            raise AssertionError("MulT-lite must pass key_padding_mask")
        self.key_padding_masks.append(key_padding_mask.detach().clone())
        return self.delegate(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )


class RecordingEncoder(nn.Module):
    """Delegate to the shared encoder while recording its inputs and padding masks."""

    def __init__(self, delegate: nn.TransformerEncoder) -> None:
        super().__init__()
        self.delegate = delegate
        self.inputs: list[torch.Tensor] = []
        self.src_key_padding_masks: list[torch.Tensor] = []

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
    ) -> torch.Tensor:
        if src_key_padding_mask is None:
            raise AssertionError("late-expert encoding must pass src_key_padding_mask")
        self.inputs.append(src.detach().clone())
        self.src_key_padding_masks.append(src_key_padding_mask.detach().clone())
        return self.delegate(
            src,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )


class CapturingCoverageGate(nn.Module):
    """Expose coverage features as fixed expert logits without new parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.inputs.append(features.detach().clone())
        return features[:, -3:]


class RecordingAvailabilityBias(nn.Module):
    """Delegate to the pooling bias while exposing its only input."""

    def __init__(self, delegate: nn.Linear) -> None:
        super().__init__()
        self.delegate = delegate
        self.inputs: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.inputs.append(features.detach().clone())
        return self.delegate(features)


class FailingPoolAttention(nn.Module):
    """Fail if masked-mean pooling evaluates learned temporal attention."""

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        raise AssertionError("masked_mean must not evaluate pool_attention")


def example_masks(*, audio_available: bool = True) -> TensorMasks:
    batch_size, positions = 2, 50
    temporal = torch.ones(batch_size, positions, dtype=torch.bool)
    return TensorMasks(
        text=temporal.clone(),
        audio=torch.full((batch_size, positions), audio_available, dtype=torch.bool),
        vision=temporal.clone(),
        temporal=temporal,
    )


def pooled_lmf_model_pair() -> tuple[MaskAwareTemporalFusion, MaskAwareTemporalFusion]:
    """Create matched gated and rank-4 LMF models with a deterministic residual."""

    torch.manual_seed(107)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    torch.manual_seed(107)
    pooled_lmf = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pooled_lmf_r4",
    ).eval()
    pooled_lmf.load_state_dict(gated.state_dict(), strict=False)
    with torch.no_grad():
        pooled_lmf.pooled_lmf_factors.copy_(torch.eye(16).repeat(3, _POOLED_LMF_RANK, 1, 1))
    return gated, pooled_lmf


def assert_same_public_output(left: object, right: object) -> None:
    assert torch.equal(left.logits, right.logits)
    assert torch.equal(left.score, right.score)
    assert torch.equal(left.gates, right.gates)
    assert torch.equal(left.temporal_attention, right.temporal_attention)
    assert left.expert_weights is right.expert_weights is None
    assert left.ordinal_logits is right.ordinal_logits is None


def availability_pooling_masks() -> TensorMasks:
    """Return varied availability including an all-unavailable temporal slot."""

    return TensorMasks(
        text=torch.tensor([[True, False, True, False, True], [False, True, False, True, False]]),
        audio=torch.tensor([[False, True, True, False, True], [False, False, True, True, False]]),
        vision=torch.tensor([[False, False, True, False, False], [True, False, False, True, False]]),
        temporal=torch.ones(2, 5, dtype=torch.bool),
    )


def test_attention_availability_adds_only_zero_bias_and_preserves_initialization_rng() -> None:
    torch.manual_seed(151)
    attention = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    attention_successor = torch.rand(5)

    torch.manual_seed(151)
    availability = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    availability_successor = torch.rand(5)

    assert availability.temporal_pooling_variant == "attention_availability"
    assert not hasattr(attention, "pool_availability_bias")
    assert availability.pool_availability_bias.weight.shape == (1, 3)
    assert torch.equal(availability.pool_availability_bias.weight, torch.zeros(1, 3))
    assert set(availability.state_dict()) - set(attention.state_dict()) == {"pool_availability_bias.weight"}
    assert all(torch.equal(attention.state_dict()[name], availability.state_dict()[name]) for name in attention.state_dict())
    assert sum(parameter.numel() for parameter in availability.parameters()) - sum(
        parameter.numel() for parameter in attention.parameters()
    ) == 3
    assert torch.equal(availability_successor, attention_successor)


@pytest.mark.parametrize("variant", ["attention", "attention_availability", "masked_mean"])
def test_temporal_pooling_variant_is_stored_after_validation(variant: str) -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant=variant,
    )

    assert model.temporal_pooling_variant == variant


def test_temporal_pooling_variant_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Atemporal_pooling_variant must be one of: attention, attention_availability, masked_mean\Z",
    ):
        MaskAwareTemporalFusion(
            hidden_size=16,
            heads=4,
            layers=1,
            dropout=0.0,
            temporal_pooling_variant="unsupported",
        )


def test_masked_mean_pooling_uses_uniform_valid_attention_without_learned_pool() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="masked_mean",
    ).eval()
    model.pool_attention = FailingPoolAttention()
    masks = availability_pooling_masks()

    output = model(
        text=torch.randn(2, 5, 768),
        audio=torch.randn(2, 5, 74),
        vision=torch.randn(2, 5, 35),
        masks=masks,
    )

    temporal = masks.temporal & (masks.text | masks.audio | masks.vision)
    expected = temporal.to(dtype=output.temporal_attention.dtype)
    expected = expected / temporal.sum(dim=1, keepdim=True).to(dtype=expected.dtype)
    assert torch.equal(output.temporal_attention, expected)


def test_masked_mean_ignores_unavailable_raw_values_and_appended_padding() -> None:
    torch.manual_seed(167)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="masked_mean",
    ).eval()
    positions, padding = 4, 2
    text = torch.randn(2, positions, 768)
    audio = torch.randn(2, positions, 74)
    vision = torch.randn(2, positions, 35)
    masks = TensorMasks(
        text=torch.tensor([[True, False, True, True], [False, True, True, True]]),
        audio=torch.tensor([[False, True, True, False], [True, False, True, True]]),
        vision=torch.tensor([[True, True, False, True], [True, True, False, False]]),
        temporal=torch.ones(2, positions, dtype=torch.bool),
    )

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    changed = model(
        text=text.masked_fill(~masks.text.unsqueeze(-1), 1_000_000.0),
        audio=audio.masked_fill(~masks.audio.unsqueeze(-1), 1_000_000.0),
        vision=vision.masked_fill(~masks.vision.unsqueeze(-1), 1_000_000.0),
        masks=masks,
    )
    assert_same_public_output(changed, baseline)

    padded_masks = TensorMasks(
        text=torch.cat((masks.text, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        audio=torch.cat((masks.audio, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        vision=torch.cat((masks.vision, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        temporal=torch.cat((masks.temporal, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
    )
    padded_model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="masked_mean",
    ).eval()
    padded_model.load_state_dict(model.state_dict())
    padded = padded_model(
        text=torch.cat((text, torch.full((2, padding, 768), 1_000_000.0)), dim=1),
        audio=torch.cat((audio, torch.full((2, padding, 74), 1_000_000.0)), dim=1),
        vision=torch.cat((vision, torch.full((2, padding, 35), 1_000_000.0)), dim=1),
        masks=padded_masks,
    )

    torch.testing.assert_close(padded.logits, baseline.logits)
    torch.testing.assert_close(padded.score, baseline.score)
    torch.testing.assert_close(padded.gates[:, :positions], baseline.gates)
    torch.testing.assert_close(padded.temporal_attention[:, :positions], baseline.temporal_attention)
    assert torch.equal(padded.gates[:, positions:], torch.zeros_like(padded.gates[:, positions:]))
    assert torch.equal(
        padded.temporal_attention[:, positions:], torch.zeros_like(padded.temporal_attention[:, positions:])
    )


def test_masked_mean_preserves_attention_state_and_rng_without_pool_attention_gradient() -> None:
    torch.manual_seed(179)
    attention = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    attention_successor = torch.rand(5)

    torch.manual_seed(179)
    masked_mean = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="masked_mean",
    )
    masked_mean_successor = torch.rand(5)

    assert masked_mean.state_dict().keys() == attention.state_dict().keys()
    assert all(torch.equal(masked_mean.state_dict()[name], attention.state_dict()[name]) for name in attention.state_dict())
    assert torch.equal(masked_mean_successor, attention_successor)

    output = masked_mean(
        text=torch.randn(2, 5, 768),
        audio=torch.randn(2, 5, 74),
        vision=torch.randn(2, 5, 35),
        masks=availability_pooling_masks(),
    )
    (output.logits.square().sum() + output.score.square().sum()).backward()

    assert masked_mean.pool_attention.weight.grad is None
    assert masked_mean.pool_attention.bias.grad is None


def test_attention_availability_pooling_rejects_late_expert_shared_fusion() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Aattention_availability temporal pooling is unsupported with late_expert_shared fusion\Z",
    ):
        MaskAwareTemporalFusion(
            hidden_size=16,
            heads=4,
            layers=1,
            dropout=0.0,
            fusion_variant="late_expert_shared",
            temporal_pooling_variant="attention_availability",
        )


def test_attention_availability_zero_bias_is_exactly_attention() -> None:
    torch.manual_seed(157)
    attention = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    torch.manual_seed(157)
    availability = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    masks = availability_pooling_masks()
    text = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 74)
    vision = torch.randn(2, 5, 35)

    assert_same_public_output(
        availability(text=text, audio=audio, vision=vision, masks=masks),
        attention(text=text, audio=audio, vision=vision, masks=masks),
    )


def test_attention_availability_bias_receives_only_floating_availability() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    recorder = RecordingAvailabilityBias(model.pool_availability_bias)
    model.pool_availability_bias = recorder
    masks = availability_pooling_masks()

    model(
        text=torch.randn(2, 5, 768),
        audio=torch.randn(2, 5, 74),
        vision=torch.randn(2, 5, 35),
        masks=masks,
    )

    assert len(recorder.inputs) == 1
    assert recorder.inputs[0].dtype.is_floating_point
    assert torch.equal(
        recorder.inputs[0],
        torch.stack((masks.text, masks.audio, masks.vision), dim=-1).to(dtype=torch.float32),
    )


def test_attention_availability_zeroes_all_invalid_attention_and_ignores_unavailable_raw_values() -> None:
    torch.manual_seed(163)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    masks = availability_pooling_masks()
    text = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 74)
    vision = torch.randn(2, 5, 35)

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    changed = model(
        text=text.masked_fill(~masks.text.unsqueeze(-1), 1_000_000.0),
        audio=audio.masked_fill(~masks.audio.unsqueeze(-1), 1_000_000.0),
        vision=vision.masked_fill(~masks.vision.unsqueeze(-1), 1_000_000.0),
        masks=masks,
    )

    invalid = masks.temporal & ~(masks.text | masks.audio | masks.vision)
    assert torch.equal(baseline.temporal_attention[invalid], torch.zeros_like(baseline.temporal_attention[invalid]))
    assert_same_public_output(changed, baseline)


def test_attention_availability_ignores_appended_fully_unavailable_padding() -> None:
    torch.manual_seed(167)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    positions, padding = 4, 2
    text = torch.randn(2, positions, 768)
    audio = torch.randn(2, positions, 74)
    vision = torch.randn(2, positions, 35)
    masks = TensorMasks(
        text=torch.tensor([[True, False, True, True], [False, True, True, True]]),
        audio=torch.tensor([[False, True, True, False], [True, False, True, True]]),
        vision=torch.tensor([[True, True, False, True], [True, True, False, False]]),
        temporal=torch.ones(2, positions, dtype=torch.bool),
    )
    padded_masks = TensorMasks(
        text=torch.cat((masks.text, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        audio=torch.cat((masks.audio, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        vision=torch.cat((masks.vision, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        temporal=torch.cat((masks.temporal, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
    )
    padded_model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    ).eval()
    padded_model.load_state_dict(model.state_dict())
    base_recorder = RecordingEncoder(model.temporal_encoder)
    padded_recorder = RecordingEncoder(padded_model.temporal_encoder)
    model.temporal_encoder = base_recorder
    padded_model.temporal_encoder = padded_recorder

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    padded = padded_model(
        text=torch.cat((text, torch.full((2, padding, 768), 1_000_000.0)), dim=1),
        audio=torch.cat((audio, torch.full((2, padding, 74), 1_000_000.0)), dim=1),
        vision=torch.cat((vision, torch.full((2, padding, 35), 1_000_000.0)), dim=1),
        masks=padded_masks,
    )

    torch.testing.assert_close(padded.logits, baseline.logits)
    torch.testing.assert_close(padded.score, baseline.score)
    torch.testing.assert_close(padded.gates[:, :positions], baseline.gates)
    torch.testing.assert_close(padded.temporal_attention[:, :positions], baseline.temporal_attention)
    assert torch.equal(padded.gates[:, positions:], torch.zeros_like(padded.gates[:, positions:]))
    assert torch.equal(
        padded.temporal_attention[:, positions:], torch.zeros_like(padded.temporal_attention[:, positions:])
    )
    torch.testing.assert_close(
        padded_recorder.inputs[0][:, :positions],
        base_recorder.inputs[0],
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(
        padded_recorder.inputs[0][:, positions:],
        torch.zeros_like(padded_recorder.inputs[0][:, positions:]),
    )


def test_attention_availability_bias_weights_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(173)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_pooling_variant="attention_availability",
    )
    masks = availability_pooling_masks()
    output = model(
        text=torch.randn(2, 5, 768),
        audio=torch.randn(2, 5, 74),
        vision=torch.randn(2, 5, 35),
        masks=masks,
    )

    (output.logits.square().sum() + output.score.square().sum()).backward()

    gradient = model.pool_availability_bias.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient != 0)


@pytest.mark.parametrize("variant", ["none", "availability_embedding"])
def test_temporal_context_variant_is_stored_after_validation(variant: str) -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_context_variant=variant,
    )

    assert model.temporal_context_variant == variant


def test_temporal_context_variant_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Atemporal_context_variant must be one of: none, availability_embedding\Z",
    ):
        MaskAwareTemporalFusion(
            hidden_size=16,
            heads=4,
            layers=1,
            dropout=0.0,
            temporal_context_variant="unsupported",
        )


def test_availability_embedding_rejects_late_expert_shared_fusion() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Aavailability_embedding temporal context is unsupported with late_expert_shared fusion\Z",
    ):
        MaskAwareTemporalFusion(
            hidden_size=16,
            heads=4,
            layers=1,
            dropout=0.0,
            fusion_variant="late_expert_shared",
            temporal_context_variant="availability_embedding",
        )


def test_availability_embedding_adds_only_zero_weight_and_preserves_rng_and_none_forward() -> None:
    torch.manual_seed(181)
    none = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    none_successor = torch.rand(5)

    torch.manual_seed(181)
    availability = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_context_variant="availability_embedding",
    ).eval()
    availability_successor = torch.rand(5)
    masks = availability_pooling_masks()
    text = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 74)
    vision = torch.randn(2, 5, 35)

    assert availability.availability_embedding.weight.shape == (16, 3)
    assert torch.equal(availability.availability_embedding.weight, torch.zeros(16, 3))
    assert set(availability.state_dict()) - set(none.state_dict()) == {"availability_embedding.weight"}
    assert all(torch.equal(none.state_dict()[name], availability.state_dict()[name]) for name in none.state_dict())
    assert sum(parameter.numel() for parameter in availability.parameters()) - sum(parameter.numel() for parameter in none.parameters()) == 48
    assert torch.equal(availability_successor, none_successor)
    assert_same_public_output(
        availability(text=text, audio=audio, vision=vision, masks=masks),
        none(text=text, audio=audio, vision=vision, masks=masks),
    )


def test_availability_embedding_ignores_unavailable_values_and_appended_padding() -> None:
    torch.manual_seed(191)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_context_variant="availability_embedding",
    ).eval()
    with torch.no_grad():
        model.availability_embedding.weight.fill_(0.25)
    positions, padding = 4, 2
    text = torch.randn(2, positions, 768)
    audio = torch.randn(2, positions, 74)
    vision = torch.randn(2, positions, 35)
    masks = TensorMasks(
        text=torch.tensor([[True, False, True, True], [False, True, True, True]]),
        audio=torch.tensor([[False, True, True, False], [True, False, True, True]]),
        vision=torch.tensor([[True, True, False, True], [True, True, False, False]]),
        temporal=torch.ones(2, positions, dtype=torch.bool),
    )

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    changed = model(
        text=text.masked_fill(~masks.text.unsqueeze(-1), 1_000_000.0),
        audio=audio.masked_fill(~masks.audio.unsqueeze(-1), 1_000_000.0),
        vision=vision.masked_fill(~masks.vision.unsqueeze(-1), 1_000_000.0),
        masks=masks,
    )
    assert_same_public_output(changed, baseline)

    padded_masks = TensorMasks(
        text=torch.cat((masks.text, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        audio=torch.cat((masks.audio, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        vision=torch.cat((masks.vision, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        temporal=torch.cat((masks.temporal, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
    )
    padded_model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_context_variant="availability_embedding",
    ).eval()
    padded_model.load_state_dict(model.state_dict())
    padded = padded_model(
        text=torch.cat((text, torch.full((2, padding, 768), 1_000_000.0)), dim=1),
        audio=torch.cat((audio, torch.full((2, padding, 74), 1_000_000.0)), dim=1),
        vision=torch.cat((vision, torch.full((2, padding, 35), 1_000_000.0)), dim=1),
        masks=padded_masks,
    )

    torch.testing.assert_close(padded.logits, baseline.logits)
    torch.testing.assert_close(padded.score, baseline.score)
    torch.testing.assert_close(padded.gates[:, :positions], baseline.gates)
    torch.testing.assert_close(padded.temporal_attention[:, :positions], baseline.temporal_attention)
    assert torch.equal(padded.gates[:, positions:], torch.zeros_like(padded.gates[:, positions:]))
    assert torch.equal(
        padded.temporal_attention[:, positions:], torch.zeros_like(padded.temporal_attention[:, positions:])
    )


def test_availability_embedding_weight_receives_finite_nonzero_gradient() -> None:
    torch.manual_seed(193)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_context_variant="availability_embedding",
    )
    output = model(
        text=torch.randn(2, 5, 768),
        audio=torch.randn(2, 5, 74),
        vision=torch.randn(2, 5, 35),
        masks=availability_pooling_masks(),
    )

    (output.logits.square().sum() + output.score.square().sum()).backward()

    gradient = model.availability_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_mask_aware_fusion_returns_three_logits_and_bounded_score() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


def test_text_anchor_residual_fusion_constructs_and_returns_valid_predictions() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="text_anchor_residual",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.logits.shape == (2, 3)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isin(output.logits.argmax(dim=1), torch.tensor([0, 1, 2])).all()


def test_pairwise_hadamard_residual_fusion_constructs_and_returns_valid_predictions() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pairwise_hadamard_residual",
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)
    masks = example_masks()

    output = model(text=text, audio=audio, vision=vision, masks=masks)

    assert output.logits.shape == (2, 3)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isin(output.logits.argmax(dim=1), torch.tensor([0, 1, 2])).all()


def test_pairwise_hadamard_residual_averages_available_products_and_zeros_padding() -> None:
    states = (
        torch.tensor([[[2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]]),
        torch.tensor([[[3.0, 5.0], [3.0, 5.0], [3.0, 5.0], [3.0, 5.0]]]),
        torch.tensor([[[7.0, 11.0], [7.0, 11.0], [7.0, 11.0], [7.0, 11.0]]]),
    )
    availability = torch.tensor([[[True, False, False], [True, True, False], [True, True, True], [True, True, True]]])
    temporal = torch.tensor([[True, True, True, False]])
    expected = torch.tensor([[[0.0, 0.0], [6.0, 20.0], [41.0 / 3.0, 119.0 / 3.0], [0.0, 0.0]]])

    torch.testing.assert_close(_pairwise_hadamard_residual(states, availability, temporal), expected)


@pytest.mark.parametrize("missing_modality", [0, 1, 2], ids=["text", "audio", "vision"])
def test_pooled_lmf_residual_pools_available_means_and_zeroes_incomplete_rows(missing_modality: int) -> None:
    states = (
        torch.tensor([[[2.0, 4.0], [4.0, 8.0], [101.0, 103.0]]]),
        torch.tensor([[[3.0, 5.0], [107.0, 109.0], [113.0, 127.0]]]),
        torch.tensor([[[7.0, 11.0], [13.0, 17.0], [131.0, 137.0]]]),
    )
    availability = torch.tensor([[[True, True, True], [True, False, True], [True, False, True]]])
    temporal = torch.tensor([[True, True, False]])
    factors = torch.eye(2).repeat(3, 4, 1, 1)
    expected = torch.tensor([[360.0, 1680.0]])

    torch.testing.assert_close(_pooled_lmf_residual(states, availability, temporal, factors), expected)

    incomplete_availability = availability.clone()
    incomplete_availability[..., missing_modality] = False
    assert torch.equal(
        _pooled_lmf_residual(states, incomplete_availability, temporal, factors),
        torch.zeros_like(expected),
    )


def test_pooled_lmf_residual_sums_distinct_rankwise_linear_transforms() -> None:
    states = (
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0]]]),
    )
    availability = torch.tensor([[[True, True, True]]])
    temporal = torch.tensor([[True]])
    factors = torch.tensor(
        [
            [
                [[1.0, 2.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 2.0]],
                [[1.0, 1.0], [1.0, -1.0]],
            ],
            [
                [[1.0, 0.0], [1.0, 1.0]],
                [[2.0, 0.0], [0.0, 1.0]],
                [[0.0, 2.0], [1.0, 0.0]],
                [[1.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[2.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 2.0]],
                [[2.0, 0.0], [0.0, 3.0]],
                [[0.0, 1.0], [2.0, 1.0]],
            ],
        ]
    )
    expected = torch.tensor([[782.0, 546.0]])

    torch.testing.assert_close(_pooled_lmf_residual(states, availability, temporal, factors), expected)


def test_pooled_lmf_r4_constructs_with_rank_factors_and_preserves_gated_initialization() -> None:
    torch.manual_seed(109)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    torch.manual_seed(109)
    pooled_lmf = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pooled_lmf_r4",
    )

    output = pooled_lmf(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert _POOLED_LMF_RANK == 4
    assert pooled_lmf.pooled_lmf_factors.shape == (3, 4, 16, 16)
    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert all(torch.equal(gated.state_dict()[name], pooled_lmf.state_dict()[name]) for name in gated.state_dict())
    assert sum(parameter.numel() for parameter in pooled_lmf.parameters()) - sum(
        parameter.numel() for parameter in gated.parameters()
    ) == 3 * 4 * 16 * 16


def test_pooled_lmf_r4_preserves_successor_rng_state() -> None:
    torch.manual_seed(113)
    MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    gated_successor = torch.rand(5)

    torch.manual_seed(113)
    MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="pooled_lmf_r4")
    pooled_lmf_successor = torch.rand(5)

    assert torch.equal(pooled_lmf_successor, gated_successor)


@pytest.mark.parametrize(
    ("available",),
    [
        ((True, False, False),),
        ((False, True, False),),
        ((False, False, True),),
        ((True, True, False),),
        ((True, False, True),),
        ((False, True, True),),
    ],
)
def test_pooled_lmf_r4_is_exactly_gated_and_has_zero_factor_gradients_without_all_modalities(
    available: tuple[bool, bool, bool],
) -> None:
    gated, pooled_lmf = pooled_lmf_model_pair()
    temporal = torch.ones(2, 4, dtype=torch.bool)
    masks = TensorMasks(
        text=torch.full_like(temporal, available[0]),
        audio=torch.full_like(temporal, available[1]),
        vision=torch.full_like(temporal, available[2]),
        temporal=temporal,
    )
    text = torch.randn(2, 4, 768)
    audio = torch.randn(2, 4, 74)
    vision = torch.randn(2, 4, 35)

    gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
    pooled_output = pooled_lmf(text=text, audio=audio, vision=vision, masks=masks)

    assert_same_public_output(pooled_output, gated_output)
    factor_gradient = torch.autograd.grad(
        pooled_output.logits.square().sum() + pooled_output.score.square().sum(),
        pooled_lmf.pooled_lmf_factors,
        allow_unused=True,
    )[0]
    if factor_gradient is not None:
        assert torch.isfinite(factor_gradient).all()
        assert torch.equal(factor_gradient, torch.zeros_like(factor_gradient))


def test_pooled_lmf_r4_keeps_incomplete_rows_gated_in_a_mixed_batch() -> None:
    gated, pooled_lmf = pooled_lmf_model_pair()
    temporal = torch.ones(3, 4, dtype=torch.bool)
    masks = TensorMasks(
        text=torch.ones(3, 4, dtype=torch.bool),
        audio=torch.tensor([[True] * 4, [False] * 4, [True] * 4]),
        vision=torch.tensor([[True] * 4, [False] * 4, [False] * 4]),
        temporal=temporal,
    )
    text = torch.randn(3, 4, 768)
    audio = torch.randn(3, 4, 74)
    vision = torch.randn(3, 4, 35)

    gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
    pooled_output = pooled_lmf(text=text, audio=audio, vision=vision, masks=masks)

    assert torch.equal(pooled_output.logits[1:], gated_output.logits[1:])
    assert torch.equal(pooled_output.score[1:], gated_output.score[1:])
    assert torch.equal(pooled_output.gates[1:], gated_output.gates[1:])
    assert torch.equal(pooled_output.temporal_attention[1:], gated_output.temporal_attention[1:])
    assert not (
        torch.equal(pooled_output.logits[:1], gated_output.logits[:1])
        and torch.equal(pooled_output.score[:1], gated_output.score[:1])
    )

    incomplete_loss = pooled_output.logits[1:].square().sum() + pooled_output.score[1:].square().sum()
    incomplete_gradient = torch.autograd.grad(
        incomplete_loss,
        pooled_lmf.pooled_lmf_factors,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if incomplete_gradient is not None:
        assert torch.isfinite(incomplete_gradient).all()
        assert torch.equal(incomplete_gradient, torch.zeros_like(incomplete_gradient))

    (pooled_output.logits.square().sum() + pooled_output.score.square().sum()).backward()
    assert pooled_lmf.pooled_lmf_factors.grad is not None
    assert torch.isfinite(pooled_lmf.pooled_lmf_factors.grad).all()
    assert pooled_lmf.pooled_lmf_factors.grad.abs().sum() > 0


def test_pooled_lmf_r4_changes_predictions_when_all_modalities_are_available() -> None:
    gated, pooled_lmf = pooled_lmf_model_pair()
    masks = TensorMasks(
        text=torch.ones(2, 4, dtype=torch.bool),
        audio=torch.ones(2, 4, dtype=torch.bool),
        vision=torch.ones(2, 4, dtype=torch.bool),
        temporal=torch.ones(2, 4, dtype=torch.bool),
    )
    text = torch.randn(2, 4, 768)
    audio = torch.randn(2, 4, 74)
    vision = torch.randn(2, 4, 35)

    with torch.no_grad():
        gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
        pooled_output = pooled_lmf(text=text, audio=audio, vision=vision, masks=masks)

    assert not (
        torch.equal(pooled_output.logits, gated_output.logits) and torch.equal(pooled_output.score, gated_output.score)
    )


def test_pooled_lmf_r4_ignores_unavailable_raw_modality_values() -> None:
    _, pooled_lmf = pooled_lmf_model_pair()
    temporal = torch.ones(2, 4, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, 1] = False
    audio_available = temporal.clone()
    audio_available[1, 2] = False
    vision_available = temporal.clone()
    vision_available[0, 3] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 4, 768)
    audio = torch.randn(2, 4, 74)
    vision = torch.randn(2, 4, 35)

    with torch.no_grad():
        baseline = pooled_lmf(text=text, audio=audio, vision=vision, masks=masks)
        changed = pooled_lmf(
            text=text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0),
            audio=audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0),
            vision=vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0),
            masks=masks,
        )

    assert_same_public_output(changed, baseline)


def test_pooled_lmf_r4_ignores_appended_temporal_padding() -> None:
    _, pooled_lmf = pooled_lmf_model_pair()
    states = tuple(torch.randn(2, 3, 16) for _ in range(3))
    availability = torch.ones(2, 3, 3, dtype=torch.bool)
    temporal = torch.ones(2, 3, dtype=torch.bool)
    padded_states = tuple(
        torch.cat((state, torch.full((2, 2, 16), 1_000_000.0)), dim=1) for state in states
    )
    padded_availability = torch.cat((availability, torch.zeros(2, 2, 3, dtype=torch.bool)), dim=1)
    padded_temporal = torch.cat((temporal, torch.zeros(2, 2, dtype=torch.bool)), dim=1)

    torch.testing.assert_close(
        _pooled_lmf_residual(states, availability, temporal, pooled_lmf.pooled_lmf_factors),
        _pooled_lmf_residual(
            padded_states,
            padded_availability,
            padded_temporal,
            pooled_lmf.pooled_lmf_factors,
        ),
    )

    masks = TensorMasks(
        text=torch.ones(2, 3, dtype=torch.bool),
        audio=torch.ones(2, 3, dtype=torch.bool),
        vision=torch.ones(2, 3, dtype=torch.bool),
        temporal=temporal,
    )
    text = torch.randn(2, 3, 768)
    audio = torch.randn(2, 3, 74)
    vision = torch.randn(2, 3, 35)
    padded_masks = TensorMasks(
        text=padded_availability[..., 0],
        audio=padded_availability[..., 1],
        vision=padded_availability[..., 2],
        temporal=padded_temporal,
    )

    with torch.no_grad():
        baseline = pooled_lmf(text=text, audio=audio, vision=vision, masks=masks)
        padded = pooled_lmf(
            text=torch.cat((text, torch.full((2, 2, 768), 1_000_000.0)), dim=1),
            audio=torch.cat((audio, torch.full((2, 2, 74), 1_000_000.0)), dim=1),
            vision=torch.cat((vision, torch.full((2, 2, 35), 1_000_000.0)), dim=1),
            masks=padded_masks,
        )

    positions = temporal.shape[1]
    assert torch.equal(padded.logits, baseline.logits)
    assert torch.equal(padded.score, baseline.score)
    assert torch.equal(padded.gates[:, :positions], baseline.gates)
    assert torch.equal(padded.temporal_attention[:, :positions], baseline.temporal_attention)
    assert torch.equal(padded.gates[:, positions:], torch.zeros_like(padded.gates[:, positions:]))
    assert torch.equal(
        padded.temporal_attention[:, positions:], torch.zeros_like(padded.temporal_attention[:, positions:])
    )


def test_pairwise_hadamard_residual_changes_predictions_with_all_modalities() -> None:
    torch.manual_seed(73)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    pairwise = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pairwise_hadamard_residual",
    ).eval()
    assert gated.state_dict().keys() == pairwise.state_dict().keys()
    assert sum(parameter.numel() for parameter in gated.parameters()) == sum(
        parameter.numel() for parameter in pairwise.parameters()
    )
    pairwise.load_state_dict(gated.state_dict(), strict=True)
    masks = example_masks()
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
        pairwise_output = pairwise(text=text, audio=audio, vision=vision, masks=masks)

    assert not (
        torch.equal(pairwise_output.logits, gated_output.logits)
        and torch.equal(pairwise_output.score, gated_output.score)
    )


def test_pairwise_hadamard_residual_is_exactly_gated_with_text_only_rows() -> None:
    torch.manual_seed(79)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    pairwise = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pairwise_hadamard_residual",
    ).eval()
    assert gated.state_dict().keys() == pairwise.state_dict().keys()
    assert sum(parameter.numel() for parameter in gated.parameters()) == sum(
        parameter.numel() for parameter in pairwise.parameters()
    )
    pairwise.load_state_dict(gated.state_dict(), strict=True)
    temporal = torch.ones(2, 50, dtype=torch.bool)
    masks = TensorMasks(
        text=temporal.clone(),
        audio=torch.zeros_like(temporal),
        vision=torch.zeros_like(temporal),
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
        pairwise_output = pairwise(text=text, audio=audio, vision=vision, masks=masks)

    assert torch.equal(pairwise_output.logits, gated_output.logits)
    assert torch.equal(pairwise_output.score, gated_output.score)
    assert torch.equal(pairwise_output.gates, gated_output.gates)
    assert torch.equal(pairwise_output.temporal_attention, gated_output.temporal_attention)
    assert pairwise_output.expert_weights is gated_output.expert_weights is None
    assert pairwise_output.ordinal_logits is gated_output.ordinal_logits is None


def test_pairwise_hadamard_residual_ignores_all_unavailable_raw_modality_values() -> None:
    torch.manual_seed(83)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="pairwise_hadamard_residual",
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, [3, 11]] = False
    text_available[1, [5, 19]] = False
    audio_available = temporal.clone()
    audio_available[0, [1, 5, 17]] = False
    audio_available[1, [2, 8, 23]] = False
    vision_available = temporal.clone()
    vision_available[0, [2, 8, 23]] = False
    vision_available[1, [1, 5, 17]] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        baseline = model(text=text, audio=audio, vision=vision, masks=masks)
        changed = model(
            text=text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0),
            audio=audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0),
            vision=vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0),
            masks=masks,
        )

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_text_anchor_residual_ignores_all_unavailable_raw_modality_values() -> None:
    torch.manual_seed(63)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="text_anchor_residual",
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, [3, 11]] = False
    text_available[1, [5, 19]] = False
    audio_available = temporal.clone()
    audio_available[0, [1, 5, 17]] = False
    audio_available[1, [2, 8, 23]] = False
    vision_available = temporal.clone()
    vision_available[0, [2, 8, 23]] = False
    vision_available[1, [1, 5, 17]] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        baseline = model(text=text, audio=audio, vision=vision, masks=masks)
        changed = model(
            text=text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0),
            audio=audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0),
            vision=vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0),
            masks=masks,
        )

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_text_anchor_residual_changes_predictions_when_text_is_available() -> None:
    torch.manual_seed(67)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    anchor = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="text_anchor_residual",
    ).eval()
    anchor.load_state_dict(gated.state_dict(), strict=True)
    masks = example_masks()
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
        anchor_output = anchor(text=text, audio=audio, vision=vision, masks=masks)

    assert not (
        torch.equal(anchor_output.logits, gated_output.logits) and torch.equal(anchor_output.score, gated_output.score)
    )


def test_text_anchor_residual_is_exactly_gated_when_all_text_is_missing() -> None:
    torch.manual_seed(71)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    anchor = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="text_anchor_residual",
    ).eval()
    anchor.load_state_dict(gated.state_dict(), strict=True)
    temporal = torch.ones(2, 50, dtype=torch.bool)
    masks = TensorMasks(
        text=torch.zeros_like(temporal),
        audio=temporal.clone(),
        vision=temporal.clone(),
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        gated_output = gated(text=text, audio=audio, vision=vision, masks=masks)
        anchor_output = anchor(text=text, audio=audio, vision=vision, masks=masks)

    assert torch.equal(anchor_output.logits, gated_output.logits)
    assert torch.equal(anchor_output.score, gated_output.score)
    assert torch.equal(anchor_output.gates, gated_output.gates)
    assert torch.equal(anchor_output.temporal_attention, gated_output.temporal_attention)
    assert anchor_output.expert_weights is gated_output.expert_weights is None
    assert anchor_output.ordinal_logits is gated_output.ordinal_logits is None


@pytest.mark.parametrize("fusion_variant", ["gated", "mag_lite", "mult_lite"])
def test_corn_classifier_returns_normalized_three_class_log_probabilities(fusion_variant: str) -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant=fusion_variant,
        classification_variant="corn",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.ordinal_logits is not None
    assert output.logits.shape == (2, 3)
    assert output.ordinal_logits.shape == (2, 2)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.ordinal_logits).all()
    assert torch.allclose(torch.exp(output.logits).sum(dim=1), torch.ones(2))
    assert torch.isin(output.logits.argmax(dim=1), torch.tensor([0, 1, 2])).all()
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


@pytest.mark.parametrize("fusion_variant", ["gated", "mag_lite", "mult_lite"])
def test_corn_classifier_ignores_unavailable_raw_modality_values(fusion_variant: str) -> None:
    torch.manual_seed(61)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant=fusion_variant,
        classification_variant="corn",
    ).eval()
    masks = example_masks(audio_available=False)
    text = torch.randn(2, 50, 768)
    vision = torch.randn(2, 50, 35)

    with torch.no_grad():
        baseline = model(text=text, audio=torch.zeros(2, 50, 74), vision=vision, masks=masks)
        changed = model(text=text, audio=torch.full((2, 50, 74), 1_000_000.0), vision=vision, masks=masks)

    assert baseline.ordinal_logits is not None
    assert changed.ordinal_logits is not None
    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.ordinal_logits, baseline.ordinal_logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_corn_classifier_rejects_shared_late_expert_fusion() -> None:
    with pytest.raises(ValueError, match="corn classification is unsupported with late_expert_shared fusion"):
        MaskAwareTemporalFusion(fusion_variant="late_expert_shared", classification_variant="corn")


def test_mask_aware_fusion_accepts_late_expert_shared_variant() -> None:
    model = MaskAwareTemporalFusion(fusion_variant="late_expert_shared")

    assert model.fusion_variant == "late_expert_shared"


def test_late_expert_shared_returns_finite_bounded_predictions_and_expert_weights() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert output.gates.shape == (2, 50, 3)
    assert output.temporal_attention.shape == (2, 50)
    assert output.expert_weights is not None
    assert output.expert_weights.shape == (2, 3)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isfinite(output.gates).all()
    assert torch.isfinite(output.temporal_attention).all()
    assert torch.isfinite(output.expert_weights).all()
    torch.testing.assert_close(output.expert_weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


def test_masked_mean_late_expert_encodes_active_rows_with_uniform_attention() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
        temporal_pooling_variant="masked_mean",
    ).eval()
    model.pool_attention = FailingPoolAttention()
    expert_mask = torch.tensor(
        [[True, False, True, True, False], [False, False, False, False, False], [False, True, False, False, True]]
    )

    representation, attention = model._encode_late_expert(torch.randn(3, 5, 16), expert_mask)

    expected = expert_mask.to(dtype=attention.dtype)
    expected[[0, 2]] = expected[[0, 2]] / expert_mask[[0, 2]].sum(dim=1, keepdim=True).to(dtype=attention.dtype)
    assert torch.equal(attention, expected)
    assert torch.equal(representation[1], torch.zeros_like(representation[1]))


def test_late_expert_shared_ignores_unavailable_raw_values_and_zeroes_inactive_experts() -> None:
    torch.manual_seed(47)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    audio_available = temporal.clone()
    audio_available[0, [1, 5, 17]] = False
    audio_available[1] = False
    vision_available = temporal.clone()
    vision_available[0] = False
    vision_available[1, [2, 8, 23]] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    audio = torch.randn(2, 50, 74).masked_fill(~audio_available.unsqueeze(-1), 0.0)
    vision = torch.randn(2, 50, 35).masked_fill(~vision_available.unsqueeze(-1), 0.0)
    changed_audio = audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0)
    changed_vision = vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0)

    with torch.no_grad():
        baseline = model(text=text, audio=audio, vision=vision, masks=masks)
        changed = model(text=text, audio=changed_audio, vision=changed_vision, masks=masks)

    assert baseline.expert_weights is not None
    assert changed.expert_weights is not None
    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)
    assert torch.equal(changed.expert_weights, baseline.expert_weights)
    assert changed.expert_weights[1, 1].item() == 0.0
    assert changed.expert_weights[0, 2].item() == 0.0
    assert torch.equal(changed.gates[~text_available, 0], torch.zeros((~text_available).sum()))
    assert torch.equal(changed.gates[~audio_available, 1], torch.zeros((~audio_available).sum()))
    assert torch.equal(changed.gates[~vision_available, 2], torch.zeros((~vision_available).sum()))


def test_late_expert_shared_coverage_keeps_valid_all_missing_frames_in_its_denominator() -> None:
    torch.manual_seed(48)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    ).eval()
    gate = CapturingCoverageGate()
    model.gate = gate
    temporal = torch.ones(1, 50, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, 30] = False
    audio_available = torch.zeros_like(temporal)
    audio_available[0, :25] = True
    vision_available = torch.zeros_like(temporal)
    vision_available[0, :10] = True
    retained_masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    excluded_temporal = temporal.clone()
    excluded_temporal[0, 30] = False
    excluded_masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=excluded_temporal,
    )
    text = torch.randn(1, 50, 768)
    audio = torch.randn(1, 50, 74)
    vision = torch.randn(1, 50, 35)

    with torch.no_grad():
        retained = model(text=text, audio=audio, vision=vision, masks=retained_masks)
        excluded = model(text=text, audio=audio, vision=vision, masks=excluded_masks)

    expected_retained_coverage = torch.tensor([[49 / 50, 25 / 50, 10 / 50]])
    expected_excluded_coverage = torch.tensor([[1.0, 25 / 49, 10 / 49]])
    assert retained.expert_weights is not None
    assert excluded.expert_weights is not None
    assert len(gate.inputs) == 2
    torch.testing.assert_close(gate.inputs[0][:, -3:], expected_retained_coverage)
    torch.testing.assert_close(gate.inputs[1][:, -3:], expected_excluded_coverage)
    torch.testing.assert_close(retained.expert_weights, torch.softmax(expected_retained_coverage, dim=1))
    torch.testing.assert_close(excluded.expert_weights, torch.softmax(expected_excluded_coverage, dim=1))
    assert not torch.equal(retained.expert_weights, excluded.expert_weights)


def test_late_expert_shared_encodes_only_rows_with_available_evidence() -> None:
    torch.manual_seed(49)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    ).eval()
    recorder = RecordingEncoder(model.temporal_encoder)
    model.temporal_encoder = recorder
    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    audio_available = temporal.clone()
    audio_available[0, [1, 5, 17]] = False
    audio_available[1] = False
    vision_available = temporal.clone()
    vision_available[0] = False
    vision_available[1, [2, 8, 23]] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )

    with torch.no_grad():
        model(
            text=torch.randn(2, 50, 768),
            audio=torch.randn(2, 50, 74),
            vision=torch.randn(2, 50, 35),
            masks=masks,
        )

    expected_masks = (~text_available, ~audio_available[0:1], ~vision_available[1:2])
    assert len(recorder.src_key_padding_masks) == 3
    for actual, expected in zip(recorder.src_key_padding_masks, expected_masks, strict=True):
        assert torch.equal(actual, expected)
        assert not actual.all(dim=1).any()


def test_late_expert_shared_routes_gradients_through_reused_modules() -> None:
    torch.manual_seed(51)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )
    (output.logits.square().mean() + output.score.square().mean()).backward()

    for module in (model.temporal_encoder, model.pool_attention, model.gate, model.classifier, model.regressor):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_late_expert_shared_preserves_gated_initialization_and_cpu_rng() -> None:
    torch.manual_seed(53)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    gated_next = torch.rand(4)

    torch.manual_seed(53)
    late_expert = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        fusion_variant="late_expert_shared",
    )
    late_expert_next = torch.rand(4)

    assert gated.state_dict().keys() == late_expert.state_dict().keys()
    assert all(torch.equal(gated.state_dict()[name], late_expert.state_dict()[name]) for name in gated.state_dict())
    assert torch.equal(gated_next, late_expert_next)


def test_houlsby_output_adapter_returns_finite_bounded_predictions_and_weights() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        text_adapter_variant="houlsby_output_b32",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert model.text_adapter_down.weight.shape == (32, 768)
    assert model.text_adapter_up.weight.shape == (768, 32)
    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert output.gates.shape == (2, 50, 3)
    assert output.temporal_attention.shape == (2, 50)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isfinite(output.gates).all()
    assert torch.isfinite(output.temporal_attention).all()
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


def test_houlsby_output_adapter_ignores_raw_text_when_text_is_unavailable() -> None:
    torch.manual_seed(31)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        text_adapter_variant="houlsby_output_b32",
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, [3, 11]] = False
    text_available[1, [5, 19]] = False
    masks = TensorMasks(
        text=text_available,
        audio=temporal.clone(),
        vision=temporal.clone(),
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768).masked_fill(~text_available.unsqueeze(-1), 0.0)
    changed_text = text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    changed = model(text=changed_text, audio=audio, vision=vision, masks=masks)

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_houlsby_output_adapter_linear_weights_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(37)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        text_adapter_variant="houlsby_output_b32",
    )

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )
    (output.logits.square().sum() + output.score.square().sum()).backward()

    for layer in (model.text_adapter_down, model.text_adapter_up):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_houlsby_output_adapter_preserves_common_initialization_and_cpu_rng() -> None:
    torch.manual_seed(41)
    identity = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
    identity_next = torch.rand(4)

    torch.manual_seed(41)
    adapter = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        text_adapter_variant="houlsby_output_b32",
    )
    adapter_next = torch.rand(4)

    assert not hasattr(identity, "text_adapter_down")
    assert not hasattr(identity, "text_adapter_up")
    common = set(identity.state_dict()) & set(adapter.state_dict())
    assert all(torch.equal(identity.state_dict()[name], adapter.state_dict()[name]) for name in common)
    assert torch.equal(identity_next, adapter_next)


def test_mask_aware_fusion_rejects_unsupported_text_adapter_variant() -> None:
    with pytest.raises(
        ValueError,
        match="text_adapter_variant must be one of: identity, houlsby_output_b32",
    ):
        MaskAwareTemporalFusion(text_adapter_variant="unsupported")


def test_gate_assigns_zero_weight_to_an_unavailable_modality() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)

    output = model(
        text=torch.ones(2, 50, 768),
        audio=torch.ones(2, 50, 74),
        vision=torch.ones(2, 50, 35),
        masks=example_masks(audio_available=False),
    )

    assert torch.equal(output.gates[:, :, 1], torch.zeros_like(output.gates[:, :, 1]))


def test_unavailable_modality_values_cannot_change_any_fusion_output() -> None:
    """Availability must cut the value path before the gate network sees it."""

    torch.manual_seed(9)
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    masks = example_masks(audio_available=False)
    text = torch.randn(2, 50, 768)
    vision = torch.randn(2, 50, 35)
    baseline = model(text=text, audio=torch.zeros(2, 50, 74), vision=vision, masks=masks)
    changed = model(text=text, audio=torch.full((2, 50, 74), 1_000_000.0), vision=vision, masks=masks)

    torch.testing.assert_close(changed.logits, baseline.logits)
    torch.testing.assert_close(changed.score, baseline.score)
    torch.testing.assert_close(changed.gates, baseline.gates)
    torch.testing.assert_close(changed.temporal_attention, baseline.temporal_attention)


def test_mag_lite_fusion_returns_finite_bounded_predictions_and_weights() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mag_lite")

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert output.gates.shape == (2, 50, 3)
    assert output.temporal_attention.shape == (2, 50)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isfinite(output.gates).all()
    assert torch.isfinite(output.temporal_attention).all()
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


def test_mag_lite_ignores_raw_audio_and_vision_when_both_are_unavailable() -> None:
    torch.manual_seed(13)
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mag_lite"
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    masks = TensorMasks(
        text=temporal.clone(),
        audio=torch.zeros_like(temporal),
        vision=torch.zeros_like(temporal),
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    baseline = model(
        text=text,
        audio=torch.zeros(2, 50, 74),
        vision=torch.zeros(2, 50, 35),
        masks=masks,
    )
    changed = model(
        text=text,
        audio=torch.full((2, 50, 74), 1_000_000.0),
        vision=torch.full((2, 50, 35), 1_000_000.0),
        masks=masks,
    )

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_mult_lite_fusion_returns_finite_bounded_predictions_and_weights() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mult_lite")

    output = model(
        text=torch.randn(2, 50, 768),
        audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35),
        masks=example_masks(),
    )

    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert output.gates.shape == (2, 50, 3)
    assert output.temporal_attention.shape == (2, 50)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isfinite(output.gates).all()
    assert torch.isfinite(output.temporal_attention).all()
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)


def test_mult_lite_skips_all_missing_nonverbal_keys_and_values() -> None:
    torch.manual_seed(23)
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mult_lite"
    ).eval()
    temporal = torch.ones(2, 50, dtype=torch.bool)
    masks = TensorMasks(
        text=temporal.clone(),
        audio=torch.zeros_like(temporal),
        vision=torch.zeros_like(temporal),
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768)
    baseline = model(
        text=text,
        audio=torch.zeros(2, 50, 74),
        vision=torch.zeros(2, 50, 35),
        masks=masks,
    )
    changed = model(
        text=text,
        audio=torch.full((2, 50, 74), 1_000_000.0),
        vision=torch.full((2, 50, 35), 1_000_000.0),
        masks=masks,
    )

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)


def test_mult_lite_partial_masks_isolate_raw_values_and_preserve_attention_gradients() -> None:
    torch.manual_seed(29)
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="mult_lite"
    )
    audio_attention = RecordingAttention(model.text_from_audio)
    vision_attention = RecordingAttention(model.text_from_vision)
    model.text_from_audio = audio_attention
    model.text_from_vision = vision_attention

    temporal = torch.ones(2, 50, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, [3, 11]] = False
    audio_available = temporal.clone()
    audio_available[0, [1, 5, 17]] = False
    audio_available[1] = False
    vision_available = temporal.clone()
    vision_available[0, [2, 8, 23]] = False
    vision_available[1] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 50, 768).masked_fill(~text_available.unsqueeze(-1), 0.0)
    audio = torch.randn(2, 50, 74).masked_fill(~audio_available.unsqueeze(-1), 0.0)
    vision = torch.randn(2, 50, 35).masked_fill(~vision_available.unsqueeze(-1), 0.0)
    changed_text = text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0)
    changed_audio = audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0)
    changed_vision = vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0)

    model.eval()
    with torch.no_grad():
        baseline = model(text=text, audio=audio, vision=vision, masks=masks)
        changed = model(text=changed_text, audio=changed_audio, vision=changed_vision, masks=masks)

    assert torch.equal(changed.logits, baseline.logits)
    assert torch.equal(changed.score, baseline.score)
    assert torch.equal(changed.gates, baseline.gates)
    assert torch.equal(changed.temporal_attention, baseline.temporal_attention)
    expected_audio_mask = ~audio_available[0:1]
    expected_vision_mask = ~vision_available[0:1]
    assert len(audio_attention.key_padding_masks) == 2
    assert len(vision_attention.key_padding_masks) == 2
    assert all(torch.equal(mask, expected_audio_mask) for mask in audio_attention.key_padding_masks)
    assert all(torch.equal(mask, expected_vision_mask) for mask in vision_attention.key_padding_masks)

    model.zero_grad()
    output = model(text=text, audio=audio, vision=vision, masks=masks)
    (output.logits.square().mean() + output.score.square().mean()).backward()

    for attention in (audio_attention, vision_attention):
        gradients = [parameter.grad for parameter in attention.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients if gradient is not None) > 0


def test_mask_aware_fusion_rejects_unsupported_fusion_variant() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"\Afusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared, "
            r"text_anchor_residual, pairwise_hadamard_residual, pooled_lmf_r4\Z"
        ),
    ):
        MaskAwareTemporalFusion(fusion_variant="unsupported")


def token_rows(*, batch_size: int = 2) -> torch.Tensor:
    """Return valid BERT ids, attention masks, and token types for a batch."""

    tokens = torch.zeros(batch_size, 3, 50, dtype=torch.int64)
    tokens[:, 0, 0] = 101
    tokens[:, 1, 0] = 1
    return tokens


def test_frozen_bert_encoder_default_uses_final_layer_without_trainable_state() -> None:
    bert = FakeBert()
    encoder = FrozenBertEncoder(bert, device=torch.device("cpu"), text_encoder_variant="last_hidden_state")

    embeddings = encoder.encode(token_rows())

    assert encoder.text_encoder_variant == "last_hidden_state"
    assert torch.equal(embeddings, torch.full((2, 50, 768), 4.0))
    assert encoder.trainable_parameters() == ()
    assert encoder.trainable_state_dict() == {}
    with pytest.raises(ValueError, match=r"\Adefault frozen BERT encoder has no trainable state\Z"):
        encoder.load_trainable_state_dict({"layer_logits": torch.zeros(4), "scale": torch.ones(())})
    assert bert.training is False
    assert bert.probe.requires_grad is False
    assert bert.probe.grad is None
    assert bert.calls is not None
    assert bert.calls[0].shape == (2, 50)
    assert bert.calls[3] is False
    assert bert.calls[4] is False


def test_frozen_bert_encoder_scalar_mix_initializes_five_values_without_rng_change() -> None:
    torch.manual_seed(211)
    _ = FakeBert()
    expected_successor = torch.rand(5)

    torch.manual_seed(211)
    encoder = FrozenBertEncoder(
        FakeBert(),
        device=torch.device("cpu"),
        text_encoder_variant="last4_scalar_mix",
    )
    actual_successor = torch.rand(5)

    assert encoder.text_encoder_variant == "last4_scalar_mix"
    assert encoder.layer_logits.shape == (4,)
    assert encoder.scale.shape == ()
    assert torch.equal(encoder.layer_logits, torch.zeros(4))
    assert torch.equal(encoder.scale, torch.ones(()))
    assert sum(parameter.numel() for parameter in encoder.trainable_parameters()) == 5
    assert torch.equal(actual_successor, expected_successor)


def test_frozen_bert_encoder_scalar_mix_returns_the_final_four_layer_mean() -> None:
    bert = FakeBert()
    encoder = FrozenBertEncoder(
        bert,
        device=torch.device("cpu"),
        text_encoder_variant="last4_scalar_mix",
    )

    embeddings = encoder.encode(token_rows())

    assert embeddings.shape == (2, 50, 768)
    assert torch.equal(embeddings, torch.full((2, 50, 768), 2.5))
    assert bert.calls is not None
    assert bert.calls[3] is False
    assert bert.calls[4] is True


def test_frozen_bert_encoder_scalar_mix_gradients_do_not_reach_bert() -> None:
    bert = FakeBert()
    encoder = FrozenBertEncoder(
        bert,
        device=torch.device("cpu"),
        text_encoder_variant="last4_scalar_mix",
    )

    embeddings = encoder.encode(token_rows())
    embeddings.square().mean().backward()

    for parameter in (encoder.layer_logits, encoder.scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for parameter in bert.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None


def test_frozen_bert_encoder_scalar_mix_strictly_round_trips_trainable_state() -> None:
    encoder = FrozenBertEncoder(
        FakeBert(),
        device=torch.device("cpu"),
        text_encoder_variant="last4_scalar_mix",
    )
    expected = {
        "layer_logits": torch.tensor([-0.4, -0.1, 0.2, 0.5]),
        "scale": torch.tensor(1.25),
    }

    encoder.load_trainable_state_dict(expected)
    state = encoder.trainable_state_dict()

    assert set(state) == {"layer_logits", "scale"}
    assert all(value.device.type == "cpu" and not value.requires_grad for value in state.values())
    assert torch.equal(state["layer_logits"], expected["layer_logits"])
    assert torch.equal(state["scale"], expected["scale"])
    state["layer_logits"].fill_(99.0)
    assert not torch.equal(encoder.layer_logits, state["layer_logits"])
    assert all("model" not in name and "probe" not in name for name in state)


@pytest.mark.parametrize(
    "state",
    (
        {"layer_logits": torch.zeros(4)},
        {"layer_logits": torch.zeros(4), "scale": torch.ones(()), "unexpected": torch.zeros(1)},
        {"layer_logits": [0.0, 0.0, 0.0, 0.0], "scale": torch.ones(())},
        {"layer_logits": torch.full((4,), float("nan")), "scale": torch.ones(())},
        {"layer_logits": torch.zeros(4), "scale": torch.tensor(float("inf"))},
        {"layer_logits": torch.zeros(3), "scale": torch.ones(())},
        {"layer_logits": torch.zeros(4), "scale": torch.ones(1)},
    ),
)
def test_frozen_bert_encoder_scalar_mix_rejects_invalid_trainable_state(state: object) -> None:
    encoder = FrozenBertEncoder(
        FakeBert(),
        device=torch.device("cpu"),
        text_encoder_variant="last4_scalar_mix",
    )

    with pytest.raises(ValueError):
        encoder.load_trainable_state_dict(state)


def test_sinusoidal_position_encoding_uses_vaswani_pairs_for_odd_hidden_width() -> None:
    positions = 3

    encoding = _sinusoidal_position_encoding(
        positions,
        5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    pair_indexes = torch.arange(3, dtype=torch.float64)
    angles = torch.arange(1, positions, dtype=torch.float64).unsqueeze(1) / torch.pow(
        torch.tensor(10000.0, dtype=torch.float64),
        2 * pair_indexes / 5,
    )
    expected = torch.empty(2, 5, dtype=torch.float64)
    expected[:, 0::2] = torch.sin(angles)
    expected[:, 1::2] = torch.cos(angles[:, :2])

    assert encoding.device == torch.device("cpu")
    assert encoding.dtype == torch.float64
    assert torch.isfinite(encoding).all()
    torch.testing.assert_close(encoding[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(encoding[1:], expected)


def test_sinusoidal_position_variant_preserves_gated_state_rng_and_none_forward() -> None:
    torch.manual_seed(131)
    none = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="none",
    ).eval()
    none_successor = torch.rand(5)

    torch.manual_seed(131)
    sinusoidal = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
    ).eval()
    sinusoidal_successor = torch.rand(5)

    assert sum(parameter.numel() for parameter in none.parameters()) == sum(
        parameter.numel() for parameter in sinusoidal.parameters()
    )
    assert list(none.state_dict()) == list(sinusoidal.state_dict())
    assert all(torch.equal(none.state_dict()[name], sinusoidal.state_dict()[name]) for name in none.state_dict())
    assert torch.equal(none_successor, sinusoidal_successor)

    torch.manual_seed(131)
    default = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()

    text = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 74)
    vision = torch.randn(2, 5, 35)
    masks = TensorMasks(
        text=torch.ones(2, 5, dtype=torch.bool),
        audio=torch.ones(2, 5, dtype=torch.bool),
        vision=torch.ones(2, 5, dtype=torch.bool),
        temporal=torch.ones(2, 5, dtype=torch.bool),
    )

    assert_same_public_output(
        default(text=text, audio=audio, vision=vision, masks=masks),
        none(text=text, audio=audio, vision=vision, masks=masks),
    )


def test_sinusoidal_positions_reach_only_observed_fused_temporal_slots() -> None:
    torch.manual_seed(137)
    gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0).eval()
    torch.manual_seed(137)
    sinusoidal = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
    ).eval()
    gated_recorder = RecordingEncoder(gated.temporal_encoder)
    sinusoidal_recorder = RecordingEncoder(sinusoidal.temporal_encoder)
    gated.temporal_encoder = gated_recorder
    sinusoidal.temporal_encoder = sinusoidal_recorder

    text_available = torch.tensor([[True, False, True, True, True]])
    audio_available = torch.tensor([[True, False, False, True, True]])
    vision_available = torch.tensor([[False, False, True, True, True]])
    temporal_mask = torch.tensor([[True, True, True, False, True]])
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal_mask,
    )
    text = torch.randn(1, 5, 768)
    audio = torch.randn(1, 5, 74)
    vision = torch.randn(1, 5, 35)

    gated(text=text, audio=audio, vision=vision, masks=masks)
    sinusoidal(text=text, audio=audio, vision=vision, masks=masks)

    temporal = temporal_mask & (text_available | audio_available | vision_available)
    expected_positions = _sinusoidal_position_encoding(
        5,
        16,
        device=text.device,
        dtype=text.dtype,
    )
    gated_input = gated_recorder.inputs[0]
    sinusoidal_input = sinusoidal_recorder.inputs[0]

    expected_input = (
        gated_input + expected_positions.unsqueeze(0) * temporal.unsqueeze(-1).to(dtype=gated_input.dtype)
    ).masked_fill(~temporal.unsqueeze(-1), 0.0)
    assert torch.equal(sinusoidal_input, expected_input)
    assert torch.equal(gated_input[0, ~temporal[0]], torch.zeros_like(gated_input[0, ~temporal[0]]))
    assert torch.equal(
        sinusoidal_input[0, ~temporal[0]], torch.zeros_like(sinusoidal_input[0, ~temporal[0]])
    )
    assert torch.equal(sinusoidal_recorder.src_key_padding_masks[0], ~temporal)


def test_sinusoidal_gated_ignores_unavailable_raw_modality_values() -> None:
    torch.manual_seed(139)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
    ).eval()
    temporal = torch.ones(2, 5, dtype=torch.bool)
    text_available = temporal.clone()
    text_available[0, 1] = False
    audio_available = temporal.clone()
    audio_available[0, 2] = False
    audio_available[1, 3] = False
    vision_available = temporal.clone()
    vision_available[1, 0] = False
    masks = TensorMasks(
        text=text_available,
        audio=audio_available,
        vision=vision_available,
        temporal=temporal,
    )
    text = torch.randn(2, 5, 768).masked_fill(~text_available.unsqueeze(-1), 0.0)
    audio = torch.randn(2, 5, 74).masked_fill(~audio_available.unsqueeze(-1), 0.0)
    vision = torch.randn(2, 5, 35).masked_fill(~vision_available.unsqueeze(-1), 0.0)

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    changed = model(
        text=text.masked_fill(~text_available.unsqueeze(-1), 1_000_000.0),
        audio=audio.masked_fill(~audio_available.unsqueeze(-1), 1_000_000.0),
        vision=vision.masked_fill(~vision_available.unsqueeze(-1), 1_000_000.0),
        masks=masks,
    )

    assert_same_public_output(changed, baseline)


def test_sinusoidal_gated_ignores_appended_fully_masked_padding() -> None:
    torch.manual_seed(149)
    model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
    ).eval()
    positions, padding = 4, 2
    text = torch.randn(2, positions, 768)
    audio = torch.randn(2, positions, 74)
    vision = torch.randn(2, positions, 35)
    masks = TensorMasks(
        text=torch.ones(2, positions, dtype=torch.bool),
        audio=torch.ones(2, positions, dtype=torch.bool),
        vision=torch.ones(2, positions, dtype=torch.bool),
        temporal=torch.ones(2, positions, dtype=torch.bool),
    )
    padded_masks = TensorMasks(
        text=torch.cat((masks.text, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        audio=torch.cat((masks.audio, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        vision=torch.cat((masks.vision, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
        temporal=torch.cat((masks.temporal, torch.zeros(2, padding, dtype=torch.bool)), dim=1),
    )
    padded_model = MaskAwareTemporalFusion(
        hidden_size=16,
        heads=4,
        layers=1,
        dropout=0.0,
        temporal_position_variant="sinusoidal",
    ).eval()
    padded_model.load_state_dict(model.state_dict())
    base_recorder = RecordingEncoder(model.temporal_encoder)
    padded_recorder = RecordingEncoder(padded_model.temporal_encoder)
    model.temporal_encoder = base_recorder
    padded_model.temporal_encoder = padded_recorder

    baseline = model(text=text, audio=audio, vision=vision, masks=masks)
    padded = padded_model(
        text=torch.cat((text, torch.zeros(2, padding, 768)), dim=1),
        audio=torch.cat((audio, torch.zeros(2, padding, 74)), dim=1),
        vision=torch.cat((vision, torch.zeros(2, padding, 35)), dim=1),
        masks=padded_masks,
    )

    torch.testing.assert_close(padded.logits, baseline.logits)
    torch.testing.assert_close(padded.score, baseline.score)
    torch.testing.assert_close(padded.gates[:, :positions], baseline.gates)
    torch.testing.assert_close(padded.temporal_attention[:, :positions], baseline.temporal_attention)
    assert torch.equal(padded.gates[:, positions:], torch.zeros_like(padded.gates[:, positions:]))
    assert torch.equal(
        padded.temporal_attention[:, positions:], torch.zeros_like(padded.temporal_attention[:, positions:])
    )
    # CPU projections can vary with sequence shape before position addition; helper prefix and padding remain exact.
    torch.testing.assert_close(
        padded_recorder.inputs[0][:, :positions],
        base_recorder.inputs[0],
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(
        padded_recorder.inputs[0][:, positions:],
        torch.zeros_like(padded_recorder.inputs[0][:, positions:]),
    )
    assert torch.equal(
        _sinusoidal_position_encoding(positions + padding, 16, device=text.device, dtype=text.dtype)[:positions],
        _sinusoidal_position_encoding(positions, 16, device=text.device, dtype=text.dtype),
    )
