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


def test_mask_aware_fusion_accepts_late_expert_shared_variant() -> None:
    model = MaskAwareTemporalFusion(fusion_variant="late_expert_shared")

    assert model.fusion_variant == "late_expert_shared"


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
