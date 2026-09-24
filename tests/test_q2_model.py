from __future__ import annotations

from types import SimpleNamespace

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


def test_gate_assigns_zero_weight_to_an_unavailable_modality() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)

    output = model(
        text=torch.ones(2, 50, 768),
        audio=torch.ones(2, 50, 74),
        vision=torch.ones(2, 50, 35),
        masks=example_masks(audio_available=False),
    )

    assert torch.equal(output.gates[:, :, 1], torch.zeros_like(output.gates[:, :, 1]))


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
