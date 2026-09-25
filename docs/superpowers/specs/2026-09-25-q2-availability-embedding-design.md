# Q2 Availability-Embedding Candidate Design

## Decision

Candidate T is an A-derived, valid-only structural trial. It changes exactly
one persisted setting:

```text
temporal_context_variant: none -> availability_embedding
```

All A settings remain fixed: aligned Attachment 2 inputs, frozen final-layer
local BERT, gated fusion, flat three-class classifier, weighted hard
cross-entropy, no synthetic training missingness, no position encoding,
attention pooling, seed `20260924`, 30 epochs, batch size 64, learning rate
`0.001`, weight decay `0.0001`, and the existing clean-valid checkpoint rule.
Attachment 2 Test must not be loaded, inspected, predicted, or used in any
decision.

The target is clean valid macro-F1 at least `0.6212527658`. A reached
`0.6165677806`, leaving `0.0046849852`. A's Neutral F1 is its weakest class
score (`0.4890109890`); this is diagnostic context only, not a basis for
threshold or class-weight tuning. Candidate T is independent of all retired
loss, fusion, pooling, adapter, position, and mask-processing treatments.

## Exact Context Semantics

The model already stacks its explicit per-slot modality availability masks as
`availability[text, audio, vision]` and supplies them to the gated fusion
network. After that network has produced its usual fused vector and before the
existing temporal Transformer, Candidate T adds a zero-initialized linear
embedding of the same explicit availability evidence:

```python
availability_embedding = nn.Linear(3, hidden_size, bias=False)
nn.init.zeros_(availability_embedding.weight)

fused = fused + availability_embedding(availability.to(dtype=fused.dtype))
fused = fused.masked_fill(~temporal.unsqueeze(-1), 0.0)
```

For the required `hidden_size=128`, this contributes exactly `3 * 128 = 384`
trainable weights, with no bias, scale, activation, dropout, loss, or masking
parameter. Its zero initialization makes all public model outputs exactly
equal to A before an optimizer step. Initialization must save and restore the
Torch RNG state so all pre-existing parameter values and successor random
draws are exactly unchanged from `temporal_context_variant="none"`.

The branch uses only the model's pre-existing Boolean availability evidence; it
does not assert that a numerical zero segment is a missing-data label. Invalid
or all-unavailable temporal positions are masked after the addition, so their
raw modality values and their availability embedding cannot perturb outputs.
The embedding receives gradients on an available differentiable batch.

`late_expert_shared` returns before the fused temporal path, so
`availability_embedding` is invalid with that fusion variant and must fail
early. Other compatibility rules remain unchanged. This restriction prevents a
persisted setting from silently doing nothing.

The premise follows the limited idea that explicit observation indicators can
carry information distinct from the observed values, as in
[GRU-D](https://arxiv.org/abs/1606.01865). It is a structural hypothesis, not
a claim that availability alone should improve E-MOSEI or that unavailable
evidence is a ground-truth missingness category.

## Persistence And Replay

New TOML configurations require `temporal_context_variant` with exactly
`none` or `availability_embedding`. It is written to `run_manifest.json` and
validated by `run_q2`, `check_q2`, and strict saved-valid reconstruction before
archive access. Historical manifests without the field reconstruct with
`none`. Strict saved-valid reconstruction restores the state dict containing
the zero-initialized embedding only when the manifest names the new variant.

## Tests And Experiment Gate

Before any real run, tests must prove:

- parser/model/runner accept exactly the two context values and reject invalid
  or late-expert combinations with exact errors before archive access;
- zero initialized Candidate T has exactly the A state dict except for its
  `[128, 3]` all-zero embedding, preserves Torch RNG succession, and produces
  exact equal public outputs for the same inputs and masks;
- unavailable raw inputs and appended padding cannot affect Candidate T
  outputs, while a differentiable available batch supplies finite, nonzero
  gradient to the embedding;
- fake run/check accept the new direct `Q2Config` without an Attachment 2 Test
  accessor, persist the variant, and strict saved-valid reconstruction accepts
  it; unsupported saved values fail before archive access and absent historical
  field defaults to `none`; and
- all focused and full suites remain green.

Only then may an ignored A-normalized TOML preflight to
`train=3395`, `valid=728`, and `attachment3=30` without output. One training
run writes `artifacts/q2-valid-availability-embedding/`; one fresh strict
saved-valid reconstruction follows. The comparison record must show that A/T
manifests differ only in `temporal_context_variant`, give clean metrics,
per-class F1, the 27-scenario mean/worst, 30 Attachment 3 predictions, and
train/valid-only scope.

If macro-F1 misses `0.6212527658`, retire this exact zero-initialized
availability embedding. Do not tune width, initialization, bias, scale,
placement, optimizer, checkpoint, seed, or combine it with retired candidates.
Test remains unassessed regardless of outcome.
