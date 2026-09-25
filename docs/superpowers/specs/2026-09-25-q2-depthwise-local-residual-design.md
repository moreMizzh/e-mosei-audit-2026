# Q2 Depthwise Local Residual Candidate U Design

## Purpose

Find one additional, independently pre-registered Q2 candidate without
accessing Attachment 2 `test`.  Candidate A remains the only sequential
control: its single fixed-seed run obtained clean valid macro-F1
`0.6165677805783404`; the acceptance gate is `0.6212527658`.  Candidate U
must change exactly one normalized training semantic from A and is not a
tuning continuation of any retired candidate B-T.

## Options Considered

1. **Selected: zero-initialized depthwise local residual.** A three-wide,
   per-hidden-channel convolution gives the existing global temporal
   Transformer a small relative-local signal.  It has 384 weights at hidden
   size 128 and starts as the exact A function.
2. **Rejected for this round: local temporal difference projection.** A fixed
   `x[t+1] - x[t-1]` followed by a zero-initialized 128-by-128 projection is
   a distinct trajectory hypothesis, but adds 16,384 weights on 3,395 training
   samples.  It remains a possible later route only after returning to A; it
   is not combined with U.
3. **Rejected: another global position, pooling, modality-context, or fusion
   treatment.** Absolute sinusoidal position, attention availability pooling,
   availability embedding, and the prior fusion routes are already retired.
   A graph/local-global architecture would add multiple mechanisms and violate
   this round's one-variable boundary.

The literature makes local convolution a reasonable *hypothesis*, not an
expected gain: [Bai et al.](https://arxiv.org/abs/1803.01271) establish
temporal convolution as a sequence-modeling baseline, while
[Howard et al.](https://arxiv.org/abs/1704.04861) motivate depthwise capacity
control.  U does not claim to reproduce either result on this small,
single-split experiment.

## Frozen Candidate U

Add a persisted field:

```toml
temporal_residual_variant = "depthwise_conv3"
```

The only supported values are `none` and `depthwise_conv3`; historical saved
manifests without this field mean `none`.  The real U configuration starts
from A, retaining `fusion_variant="gated"`,
`temporal_context_variant="none"`, `temporal_position_variant="none"`,
`temporal_pooling_variant="attention"`, flat hard CE, frozen last-layer BERT,
and `synthetic_missingness_enabled=false`.  Its normalized manifest may differ
from A only in `temporal_residual_variant`.

`depthwise_conv3` constructs exactly one module, while restoring Torch RNG
after construction:

```python
self.local_depthwise_residual = nn.Conv1d(
    hidden_size,
    hidden_size,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=hidden_size,
    bias=False,
)
nn.init.zeros_(self.local_depthwise_residual.weight)
```

For `hidden_size=128`, the sole new parameter is
`local_depthwise_residual.weight` with shape `[128, 1, 3]` and 384 weights.
There is no activation, normalization, gate, scale, pointwise convolution, or
change to the loss, optimizer, seed, epoch count, checkpoint rule, data, or
mask generation.

After the existing gated fused sum and before pairwise residuals, positional
features, and the temporal Transformer, U executes:

```python
fused = fused.masked_fill(~temporal.unsqueeze(-1), 0.0)
if self.temporal_residual_variant == "depthwise_conv3":
    local = self.local_depthwise_residual(fused.transpose(1, 2)).transpose(1, 2)
    fused = (fused + local).masked_fill(~temporal.unsqueeze(-1), 0.0)
```

The first mask is mandatory: otherwise a padded or unavailable slot could
become a convolutional neighbour of a valid slot.  The second prevents a local
neighbour from generating a nonzero value at an invalid slot.  The variant is
invalid with `late_expert_shared`, whose forward path returns before there is a
fused sequence; it fails before archive/output I/O with:

```text
depthwise_conv3 temporal residual is unsupported with late_expert_shared fusion
```

## Evidence And Safety Contracts

Before the real run, tests must prove all of the following.

- New configurations require and validate the field; old manifests default to
  `none`; invalid saved semantics fail before archive access.
- Relative to `none`, U adds only the all-zero `[H,1,3]` weight.  Common state,
  successor Torch RNG values, and all public outputs are exactly equal before
  optimization.
- With a nonzero test kernel, unavailable raw feature values, internal absent
  slots, and appended padding cannot affect any valid public output.  The
  input sent to the Transformer remains zero at invalid positions.
- An available batch yields finite, nonzero convolution-weight gradients.
- `check_q2`, fake `run_q2`, and strict saved-valid replay persist and restore
  U without opening an inaccessible Attachment 2 Test sentinel.  Replay keeps
  `load_state_dict(..., strict=True)`.

The candidate's ignored configuration must retain A's fixed values: seed
`20260924`, 30 epochs, batch size 64, learning rate `0.001`, weight decay
`0.0001`, hidden size 128, 4 heads, 2 layers, dropout `0.1`, regression weight
`0.5`, and class-weight exponent `1.0`.  A preflight must produce exactly
`train=3395`, `valid=728`, and `attachment3=30` without creating an output.

One successful preflight is followed by exactly one training run and one
strict valid reconstruction.  Required artifact checks are four equal saved
and reconstructed clean metrics, support 728, 27 scenario rows, 30 unlabeled
Attachment 3 predictions, a nonempty model, and a sole normalized manifest
diff of `temporal_residual_variant: none -> depthwise_conv3`.  Attachment 2
Test is never loaded, evaluated, selected on, or reported for U.

Accept U only if clean valid macro-F1 is at least `0.6212527658`.  Otherwise
retire this exact kernel, padding, initialization, placement, bias, scale,
optimizer, checkpoint, and seed without a retry, sweep, or combination.

## Record Integrity

Before recording U, add the missing historical Candidate Q
`neutral_gate_polarity` entry to the exploration portfolio so its retirement
matches the README scoreboard.  Its artifact documents a retired macro-F1 of
`0.5967240265878949`; the current runtime intentionally supports only
`flat` and `corn`, so Q is an archival result rather than a current strict
replay contract.  U does not restore, reuse, or combine Q.
