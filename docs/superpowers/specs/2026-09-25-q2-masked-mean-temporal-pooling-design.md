# Q2 Masked-Mean Temporal-Pooling Candidate Design

## Decision

Candidate S is a valid-only, fixed-seed structural ablation of candidate A. It
changes exactly one persisted setting:

```text
temporal_pooling_variant: attention -> masked_mean
```

All other A settings remain unchanged: aligned Attachment 2 features, frozen
local BERT final layer, `gated` fusion, flat three-class head, identity text
adapter, hard weighted cross-entropy, no temporal position feature, no
synthetic training missingness, seed `20260924`, 30 epochs, batch size 64,
learning rate `0.001`, weight decay `0.0001`, and the existing clean-valid
checkpoint rule. Attachment 2 `test` must not be loaded, inspected, predicted,
or used in any decision.

The target is clean valid macro-F1 at least `0.6212527658`. A reached
`0.6165677806`; the closest retired candidate, frozen output adapter `b32`,
reached `0.6180331930`. Candidate S is independent of all retired loss,
fusion, encoder, and availability-conditioned pooling changes. It must not be
combined with them.

## Exact Pooling Semantics

After availability-masked fusion and temporal encoding, let `temporal` be the
existing Boolean mask that is true precisely where the sample has both an
in-range temporal slot and at least one available modality. Every accepted
sample already has at least one such slot. Candidate S defines:

```python
temporal_attention = temporal.to(encoded.dtype)
temporal_attention = temporal_attention / temporal_attention.sum(dim=1, keepdim=True)
pooled = torch.sum(temporal_attention.unsqueeze(-1) * encoded, dim=1)
```

Thus every valid time slot has weight `1 / valid_count`; padding and positions
with no available modality have exactly zero attention. It removes learned
content-based time selection but changes neither the temporal encoder nor the
masked fusion path. The same rule applies to the active rows of the shared
late-expert encoder so the persisted variant has one consistent meaning.

`pool_attention` remains constructed and serialized under `masked_mean`. This
preserves the default model's parameter layout, random-number consumption, and
state-dict compatibility. It is intentionally not evaluated in a masked-mean
forward pass and therefore must receive no gradient from that pass. No new
parameter, trainable scalar, loss term, threshold, checkpoint rule, or mask
meaning is introduced.

## Persistence And Replay

`masked_mean` is a third exact `temporal_pooling_variant` value. New TOML
configurations accept it, `run_manifest.json` records it through the existing
training field, and strict saved-valid reconstruction validates and restores it
before reading the archive. Historical manifests without the field still mean
the original `attention` implementation.

The existing `attention_availability` remains unchanged and incompatible with
`late_expert_shared`; masked mean is not an availability-bias variant, so it
does not add that restriction. All existing non-masked-mean behaviors must
remain bit-for-bit unchanged.

## Tests And Experiment Gate

Before any real run, tests must prove:

- parser and model validation accept exactly `attention`,
  `attention_availability`, and `masked_mean`, while an unsupported value fails
  with the updated exact message;
- masked-mean attention is uniform over true temporal positions, with exact
  zeros elsewhere, and changing unavailable raw inputs or appended padding
  cannot affect public outputs;
- construction with `masked_mean` consumes the same RNG, creates the same
  state-dict keys and values as `attention`, and leaves `pool_attention` with
  no gradient after a differentiable forward loss;
- fake-archive training records the new value without accessing the Attachment
  2 test sentinel, and strict saved-valid reconstruction uses the persisted
  value while unsupported saved values fail before archive access; and
- all focused and full suites remain green.

After a passing preflight reports exactly `train=3395`, `valid=728`, and
`attachment3=30` with no output created, Candidate S trains once into
`artifacts/q2-valid-masked-mean/`. A separate strict saved-valid report is made
once. The comparison record must prove that normalized A/S manifests differ
only in `temporal_pooling_variant`, record clean metrics, per-class F1,
27-scenario mean/worst, 30 Attachment 3 predictions, and the explicit
train/valid-only scope.

If Candidate S misses `0.6212527658`, retire this exact pooling semantics. Do
not tune pooling weights, masks, optimizer, checkpoint, seed, or combine it
with a retired candidate. Test remains unassessed regardless of outcome.
