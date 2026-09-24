"""Mask-aware temporal fusion model for Problem 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from e_mosei_audit.q2.config import (
    validate_classification_variant,
    validate_fusion_variant,
    validate_text_adapter_variant,
)


_POOLED_LMF_RANK = 4


@dataclass(frozen=True)
class TensorMasks:
    """Boolean availability masks on the same device as model inputs."""

    text: torch.Tensor
    audio: torch.Tensor
    vision: torch.Tensor
    temporal: torch.Tensor


@dataclass(frozen=True)
class Q2Output:
    """Joint prediction and inspectable fusion weights for a mini-batch."""

    logits: torch.Tensor
    score: torch.Tensor
    gates: torch.Tensor
    temporal_attention: torch.Tensor
    expert_weights: torch.Tensor | None = None
    ordinal_logits: torch.Tensor | None = None


class FrozenBertEncoder:
    """Offline BERT feature encoder shared by Attachment 2 and Attachment 3."""

    def __init__(self, model: nn.Module, *, device: torch.device) -> None:
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_local(cls, model_path: Path, *, device: torch.device) -> "FrozenBertEncoder":
        """Load the required local BERT cache without any network fallback."""

        try:
            from transformers import BertModel
        except ImportError as error:  # pragma: no cover - exercised by runtime dependency preflight.
            raise RuntimeError("Problem 2 requires transformers; install the q2 extra") from error
        model = BertModel.from_pretrained(str(model_path), local_files_only=True)
        if getattr(model.config, "hidden_size", None) != 768:
            raise ValueError("local BERT model must emit hidden size 768")
        return cls(model, device=device)

    def encode(self, token_rows: torch.Tensor) -> torch.Tensor:
        """Encode BERT token ids, attention mask, and token types from `[B, 3, 50]`."""

        if token_rows.ndim != 3 or token_rows.shape[1:] != (3, 50):
            raise ValueError("text_bert tensor must have shape [B, 3, 50]")
        if token_rows.dtype != torch.int64:
            raise ValueError("text_bert tensor must use int64 token values")
        values = token_rows.to(self.device)
        with torch.no_grad():
            output: Any = self.model(
                input_ids=values[:, 0, :],
                attention_mask=values[:, 1, :],
                token_type_ids=values[:, 2, :],
            )
        embeddings = output.last_hidden_state
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (values.shape[0], 50, 768):
            raise ValueError("BERT encoder must return [B, 50, 768] last_hidden_state")
        return embeddings


class MaskAwareTemporalFusion(nn.Module):
    """Fuse three aligned modalities without allowing unavailable evidence to leak."""

    def __init__(
        self,
        *,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        fusion_variant: str = "gated",
        text_adapter_variant: str = "identity",
        classification_variant: str = "flat",
    ) -> None:
        super().__init__()
        if hidden_size < 1 or heads < 1 or layers < 1 or hidden_size % heads:
            raise ValueError("hidden_size must be positive, layers/heads positive, and hidden_size divisible by heads")
        self.fusion_variant = validate_fusion_variant(fusion_variant)
        self.text_adapter_variant = validate_text_adapter_variant(text_adapter_variant)
        self.classification_variant = validate_classification_variant(classification_variant)
        if self.fusion_variant == "late_expert_shared" and self.classification_variant == "corn":
            raise ValueError("corn classification is unsupported with late_expert_shared fusion")
        self.text_projection = _projection(768, hidden_size)
        self.audio_projection = _projection(74, hidden_size)
        self.vision_projection = _projection(35, hidden_size)
        if self.fusion_variant == "mag_lite":
            self.mag_shift = nn.Sequential(nn.Linear(hidden_size * 3 + 2, hidden_size), nn.Tanh())
            self.mag_gate = nn.Sequential(nn.Linear(hidden_size * 3 + 2, hidden_size), nn.Sigmoid())
        if self.fusion_variant == "mult_lite":
            self.text_from_audio = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
            self.text_from_vision = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
            self.mult_lite_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.pool_attention = nn.Linear(hidden_size, 1)
        classifier_size = 2 if self.classification_variant == "corn" else 3
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size + 3), nn.Linear(hidden_size + 3, classifier_size))
        self.regressor = nn.Sequential(nn.LayerNorm(hidden_size + 3), nn.Linear(hidden_size + 3, 1))
        if self.text_adapter_variant == "houlsby_output_b32":
            rng_state = torch.get_rng_state()
            try:
                self.text_adapter_down = nn.Linear(768, 32)
                self.text_adapter_up = nn.Linear(32, 768)
                nn.init.trunc_normal_(self.text_adapter_down.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
                nn.init.zeros_(self.text_adapter_down.bias)
                nn.init.trunc_normal_(self.text_adapter_up.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
                nn.init.zeros_(self.text_adapter_up.bias)
            finally:
                torch.set_rng_state(rng_state)
        if self.fusion_variant == "pooled_lmf_r4":
            rng_state = torch.get_rng_state()
            try:
                self.pooled_lmf_factors = nn.Parameter(torch.empty(3, _POOLED_LMF_RANK, hidden_size, hidden_size))
                for modality_index in range(3):
                    for rank_index in range(_POOLED_LMF_RANK):
                        nn.init.xavier_uniform_(self.pooled_lmf_factors[modality_index, rank_index])
            finally:
                torch.set_rng_state(rng_state)

    def forward(self, *, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor, masks: TensorMasks) -> Q2Output:
        """Return joint predictions while applying masks before fusion and pooling."""

        _validate_inputs(text, audio, vision, masks)
        if self.text_adapter_variant == "houlsby_output_b32":
            text_mask = masks.text.unsqueeze(-1)
            masked_text = text.masked_fill(~text_mask, 0.0)
            delta = self.text_adapter_up(torch.relu(self.text_adapter_down(masked_text)))
            delta = delta.masked_fill(~text_mask, 0.0)
            text = (masked_text + delta).masked_fill(~text_mask, 0.0)
        availability = torch.stack((masks.text, masks.audio, masks.vision), dim=-1)
        projected_states = (
            self.text_projection(text),
            self.audio_projection(audio),
            self.vision_projection(vision),
        )
        # Remove both source values and projection biases before gating so an
        # unavailable modality cannot perturb weights of available modalities.
        states = tuple(
            state.masked_fill(~availability[..., index : index + 1], 0.0)
            for index, state in enumerate(projected_states)
        )
        any_available = availability.any(dim=-1)
        temporal = masks.temporal & any_available
        if not bool(temporal.any(dim=1).all()):
            raise ValueError("each sample must retain at least one observed temporal position")

        if self.fusion_variant == "late_expert_shared":
            return self._forward_late_expert_shared(states, availability, masks.temporal)

        if self.fusion_variant == "mag_lite":
            mag_features = torch.cat((*states, availability[..., 1:].to(dtype=text.dtype)), dim=-1)
            nonverbal_available = availability[..., 1:].any(dim=-1, keepdim=True)
            text_update = states[0] + self.mag_gate(mag_features) * self.mag_shift(mag_features)
            text_state = torch.where(
                availability[..., 0:1] & nonverbal_available,
                text_update,
                states[0],
            )
            states = (text_state, states[1], states[2])

        if self.fusion_variant == "mult_lite":
            audio_context = self._cross_attention_context(
                query=states[0],
                source=states[1],
                query_available=availability[..., 0],
                source_available=availability[..., 1],
                attention=self.text_from_audio,
            )
            vision_context = self._cross_attention_context(
                query=states[0],
                source=states[2],
                query_available=availability[..., 0],
                source_available=availability[..., 2],
                attention=self.text_from_vision,
            )
            nonverbal_present = availability[..., 1:].any(dim=1)
            source_count = nonverbal_present.sum(dim=-1).clamp_min(1).to(dtype=text.dtype).view(-1, 1, 1)
            text_update = self.mult_lite_norm(states[0] + (audio_context + vision_context) / source_count)
            can_update_text = availability[..., 0:1] & nonverbal_present.any(dim=-1).view(-1, 1, 1)
            states = (torch.where(can_update_text, text_update, states[0]), states[1], states[2])

        gate_features = torch.cat((*states, availability.to(dtype=text.dtype)), dim=-1)
        gate_logits = self.gate(gate_features).masked_fill(~availability, float("-inf"))
        safe_gate_logits = torch.where(any_available.unsqueeze(-1), gate_logits, torch.zeros_like(gate_logits))
        gates = torch.softmax(safe_gate_logits, dim=-1)
        gates = torch.where(any_available.unsqueeze(-1), gates, torch.zeros_like(gates))
        fused = sum(gates[..., index : index + 1] * state for index, state in enumerate(states))
        fused = fused.masked_fill(~temporal.unsqueeze(-1), 0.0)
        if self.fusion_variant == "pairwise_hadamard_residual":
            fused = fused + _pairwise_hadamard_residual(states, availability, masks.temporal)

        encoded = self.temporal_encoder(fused, src_key_padding_mask=~temporal)
        encoded = encoded.masked_fill(~temporal.unsqueeze(-1), 0.0)
        attention_logits = self.pool_attention(encoded).squeeze(-1).masked_fill(~temporal, float("-inf"))
        temporal_attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=1)
        if self.fusion_variant == "pooled_lmf_r4":
            residual = _pooled_lmf_residual(states, availability, masks.temporal, self.pooled_lmf_factors)
            complete_modalities = torch.stack(
                tuple((masks.temporal & availability[..., index]).any(dim=1) for index in range(3)), dim=1
            ).all(dim=1, keepdim=True)
            pooled = torch.where(complete_modalities, pooled + residual, pooled)
        availability_fraction = (availability & temporal.unsqueeze(-1)).sum(dim=1).to(text.dtype)
        availability_fraction = availability_fraction / temporal.sum(dim=1, keepdim=True).to(text.dtype)
        representation = torch.cat((pooled, availability_fraction), dim=1)
        if self.fusion_variant == "text_anchor_residual":
            text_temporal = masks.temporal & availability[..., 0]
            pooled_text, _ = self._encode_late_expert(states[0], text_temporal)
            text_coverage = text_temporal.sum(dim=1, keepdim=True).to(text.dtype)
            text_coverage = text_coverage / masks.temporal.sum(dim=1, keepdim=True).to(text.dtype)
            text_anchor = torch.cat(
                (pooled_text, text_coverage, text_coverage.new_zeros((text.shape[0], 2))), dim=1
            )
            representation = representation + text_anchor
        logits, ordinal_logits = self._classify(representation)
        return Q2Output(
            logits=logits,
            score=3.0 * torch.tanh(self.regressor(representation).squeeze(-1)),
            gates=gates,
            temporal_attention=temporal_attention,
            ordinal_logits=ordinal_logits,
        )

    def _classify(self, representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return flat logits or CORN's derived three-class log-probabilities."""

        classification = self.classifier(representation)
        if self.classification_variant == "flat":
            return classification, None
        conditional = torch.sigmoid(classification)
        probabilities = torch.stack(
            (
                1.0 - conditional[:, 0],
                conditional[:, 0] * (1.0 - conditional[:, 1]),
                conditional[:, 0] * conditional[:, 1],
            ),
            dim=1,
        )
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        return logits, classification

    def _forward_late_expert_shared(
        self,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        availability: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> Q2Output:
        """Encode each modality with the common temporal path before late fusion."""

        expert_masks = tuple(temporal_mask & availability[..., index] for index in range(3))
        encoded_experts = tuple(
            self._encode_late_expert(state, expert_mask)
            for state, expert_mask in zip(states, expert_masks, strict=True)
        )
        representations = torch.stack(tuple(result[0] for result in encoded_experts), dim=1)
        attentions = torch.stack(tuple(result[1] for result in encoded_experts), dim=1)
        temporal_count = temporal_mask.sum(dim=1, keepdim=True).to(dtype=representations.dtype)
        coverage = torch.stack(expert_masks, dim=-1).sum(dim=1).to(dtype=representations.dtype) / temporal_count
        modality_features = coverage.unsqueeze(-1) * torch.eye(
            3,
            device=representations.device,
            dtype=representations.dtype,
        )
        expert_features = torch.cat((representations, modality_features), dim=-1)
        expert_active = torch.stack(expert_masks, dim=-1).any(dim=1)
        expert_logits = self.classifier(expert_features).masked_fill(~expert_active.unsqueeze(-1), 0.0)
        expert_scores = (3.0 * torch.tanh(self.regressor(expert_features).squeeze(-1))).masked_fill(
            ~expert_active,
            0.0,
        )
        gate_features = torch.cat((representations.flatten(start_dim=1), coverage), dim=1)
        gate_logits = self.gate(gate_features).masked_fill(~expert_active, float("-inf"))
        expert_weights = torch.softmax(gate_logits, dim=1)
        logits = torch.sum(expert_weights.unsqueeze(-1) * expert_logits, dim=1)
        score = torch.sum(expert_weights * expert_scores, dim=1)
        applied_gates = expert_weights.unsqueeze(1) * torch.stack(expert_masks, dim=-1).to(
            dtype=representations.dtype
        )
        temporal_attention = torch.sum(expert_weights.unsqueeze(-1) * attentions, dim=1)
        return Q2Output(logits, score, applied_gates, temporal_attention, expert_weights)

    def _encode_late_expert(self, state: torch.Tensor, expert_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only active sample rows so Transformer never receives all-padding input."""

        batch_size, positions, hidden_size = state.shape
        representation = state.new_zeros((batch_size, hidden_size))
        attention = state.new_zeros((batch_size, positions))
        active_rows = expert_mask.any(dim=1)
        indexes = active_rows.nonzero(as_tuple=False).squeeze(1)
        if indexes.numel() == 0:
            return representation, attention
        active_mask = expert_mask.index_select(0, indexes)
        active_state = state.index_select(0, indexes).masked_fill(~active_mask.unsqueeze(-1), 0.0)
        encoded = self.temporal_encoder(active_state, src_key_padding_mask=~active_mask)
        encoded = encoded.masked_fill(~active_mask.unsqueeze(-1), 0.0)
        attention_logits = self.pool_attention(encoded).squeeze(-1).masked_fill(~active_mask, float("-inf"))
        active_attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(active_attention.unsqueeze(-1) * encoded, dim=1)
        return representation.index_copy(0, indexes, pooled), attention.index_copy(0, indexes, active_attention)

    def _cross_attention_context(
        self,
        *,
        query: torch.Tensor,
        source: torch.Tensor,
        query_available: torch.Tensor,
        source_available: torch.Tensor,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        """Attend only samples with at least one available query and source position."""

        active = query_available.any(dim=1) & source_available.any(dim=1)
        context = torch.zeros_like(query)
        indexes = active.nonzero(as_tuple=False).squeeze(1)
        if indexes.numel() == 0:
            return context
        attended, _ = attention(
            query[indexes],
            source[indexes],
            source[indexes],
            key_padding_mask=~source_available[indexes],
            need_weights=False,
        )
        attended = attended.masked_fill(~query_available[indexes].unsqueeze(-1), 0.0)
        return context.index_copy(0, indexes, attended)


def _projection(input_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())


def _pooled_lmf_residual(
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    availability: torch.Tensor,
    temporal_mask: torch.Tensor,
    factors: torch.Tensor,
) -> torch.Tensor:
    """Return a rankwise product of modality means when every modality is present."""

    hidden_size = states[0].shape[-1]
    if factors.shape != (3, _POOLED_LMF_RANK, hidden_size, hidden_size):
        raise ValueError(f"factors must have shape [3, {_POOLED_LMF_RANK}, {hidden_size}, {hidden_size}]")
    modality_masks = tuple(temporal_mask & availability[..., index] for index in range(3))
    counts = torch.stack(tuple(mask.sum(dim=1) for mask in modality_masks), dim=1)
    pooled = torch.stack(
        tuple(
            (state * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
            for state, mask, count in zip(states, modality_masks, counts.unbind(dim=1), strict=True)
        ),
        dim=1,
    )
    rankwise = torch.einsum("bmd,mrdh->bmrh", pooled, factors)
    residual = rankwise[:, 0] * rankwise[:, 1] * rankwise[:, 2]
    residual = residual.sum(dim=1)
    return torch.where(counts.gt(0).all(dim=1, keepdim=True), residual, torch.zeros_like(residual))


def _pairwise_hadamard_residual(
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    availability: torch.Tensor,
    temporal_mask: torch.Tensor,
) -> torch.Tensor:
    pair_masks = (
        temporal_mask & availability[..., 0] & availability[..., 1],
        temporal_mask & availability[..., 0] & availability[..., 2],
        temporal_mask & availability[..., 1] & availability[..., 2],
    )
    pair_products = (states[0] * states[1], states[0] * states[2], states[1] * states[2])
    residual = sum(mask.unsqueeze(-1) * product for mask, product in zip(pair_masks, pair_products, strict=True))
    pair_count = sum(mask.to(dtype=states[0].dtype) for mask in pair_masks)
    return residual / pair_count.clamp_min(1).unsqueeze(-1)


def _validate_inputs(text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor, masks: TensorMasks) -> None:
    if text.ndim != 3 or text.shape[2] != 768:
        raise ValueError("text must have shape [B, T, 768]")
    batch_size, positions, _ = text.shape
    if audio.shape != (batch_size, positions, 74):
        raise ValueError("audio must have shape [B, T, 74]")
    if vision.shape != (batch_size, positions, 35):
        raise ValueError("vision must have shape [B, T, 35]")
    expected_shape = (batch_size, positions)
    for name, value in (("text", masks.text), ("audio", masks.audio), ("vision", masks.vision), ("temporal", masks.temporal)):
        if value.shape != expected_shape or value.dtype != torch.bool or value.device != text.device:
            raise ValueError(f"{name} mask must be bool [B, T] on the text device")
