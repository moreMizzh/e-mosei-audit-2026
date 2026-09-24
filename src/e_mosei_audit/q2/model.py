"""Mask-aware temporal fusion model for Problem 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from e_mosei_audit.q2.config import validate_fusion_variant, validate_text_adapter_variant


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
    ) -> None:
        super().__init__()
        if hidden_size < 1 or heads < 1 or layers < 1 or hidden_size % heads:
            raise ValueError("hidden_size must be positive, layers/heads positive, and hidden_size divisible by heads")
        self.fusion_variant = validate_fusion_variant(fusion_variant)
        self.text_adapter_variant = validate_text_adapter_variant(text_adapter_variant)
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
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size + 3), nn.Linear(hidden_size + 3, 3))
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

        encoded = self.temporal_encoder(fused, src_key_padding_mask=~temporal)
        encoded = encoded.masked_fill(~temporal.unsqueeze(-1), 0.0)
        attention_logits = self.pool_attention(encoded).squeeze(-1).masked_fill(~temporal, float("-inf"))
        temporal_attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=1)
        availability_fraction = (availability & temporal.unsqueeze(-1)).sum(dim=1).to(text.dtype)
        availability_fraction = availability_fraction / temporal.sum(dim=1, keepdim=True).to(text.dtype)
        representation = torch.cat((pooled, availability_fraction), dim=1)
        return Q2Output(
            logits=self.classifier(representation),
            score=3.0 * torch.tanh(self.regressor(representation).squeeze(-1)),
            gates=gates,
            temporal_attention=temporal_attention,
        )

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
