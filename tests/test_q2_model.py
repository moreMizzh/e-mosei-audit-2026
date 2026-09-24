from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from e_mosei_audit.q2.model import FrozenBertEncoder, MaskAwareTemporalFusion, TensorMasks


class FakeBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.ones(1))
        self.calls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> SimpleNamespace:
        self.calls = (input_ids, attention_mask, token_type_ids, torch.is_grad_enabled())
        return SimpleNamespace(last_hidden_state=torch.ones(input_ids.shape[0], input_ids.shape[1], 768))


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
    """Delegate to the shared encoder while recording late-expert padding masks."""

    def __init__(self, delegate: nn.TransformerEncoder) -> None:
        super().__init__()
        self.delegate = delegate
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


def example_masks(*, audio_available: bool = True) -> TensorMasks:
    batch_size, positions = 2, 50
    temporal = torch.ones(batch_size, positions, dtype=torch.bool)
    return TensorMasks(
        text=temporal.clone(),
        audio=torch.full((batch_size, positions), audio_available, dtype=torch.bool),
        vision=temporal.clone(),
        temporal=temporal,
    )


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
        match=r"\Afusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared\Z",
    ):
        MaskAwareTemporalFusion(fusion_variant="unsupported")


def test_frozen_bert_encoder_uses_three_token_rows_without_gradients() -> None:
    bert = FakeBert()
    encoder = FrozenBertEncoder(bert, device=torch.device("cpu"))
    tokens = torch.zeros(2, 3, 50, dtype=torch.int64)
    tokens[:, 0, 0] = 101
    tokens[:, 1, 0] = 1

    embeddings = encoder.encode(tokens)

    assert embeddings.shape == (2, 50, 768)
    assert bert.training is False
    assert bert.probe.requires_grad is False
    assert bert.calls is not None
    assert bert.calls[0].shape == (2, 50)
    assert bert.calls[3] is False
