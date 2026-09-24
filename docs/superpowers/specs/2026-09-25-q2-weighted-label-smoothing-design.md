# Q2 Weighted Label-Smoothing Candidate Design

## Decision

Candidate Q is an A-derived, valid-only trial of fixed weighted label smoothing.
It changes exactly one persisted training semantic:

```text
classification_loss_variant: hard_ce -> weighted_label_smoothing_005
```

All other A settings remain unchanged: aligned Attachment 2 features, frozen local
BERT final layer, `gated` fusion, flat three-class head, no synthetic training
missingness, seed `20260924`, 30 epochs, batch size 64, learning rate `0.001`,
weight decay `0.0001`, existing class-weight exponent `1.0`, and the current clean
valid checkpoint rule. Attachment 2 `test` must not be loaded, inspected,
predicted, or used in any decision.

The target is clean valid macro-F1 at least `0.6212527658`. A's clean valid
macro-F1 is `0.6165677806`; the closest retired candidate, frozen output adapter
`b32`, reached `0.6180331930` but cannot be retuned. Candidate Q is independent
of every retired model, fusion, encoder, pooling, and R-Drop variant.

## Alternatives Considered

1. A larger or smaller output adapter bottleneck would follow the one positive
   adapter result, but F's retirement record explicitly prohibits bottleneck or
   other adapter hyperparameter tuning after its fixed `b32` trial.
2. Replacing learned modality fusion or temporal attention with a uniform mean is
   a valid future zero-parameter structural ablation, but it introduces a new
   representation path without prior registration and has less direct evidence for
   the observed small-data classifier behavior.
3. Fixed weighted label smoothing is selected. It was retained explicitly as a
   future independent candidate in the R-Drop design, has one fixed value rather
   than a sweep, and leaves the representation/inference path intact. Label
   smoothing replaces a hard target with a mixture of that target and the uniform
   class distribution; it can reduce overconfidence and improve generalization or
   calibration, though no paper guarantees an E-MOSEI gain. [Muller, Kornblith,
   and Hinton (2019)](https://arxiv.org/abs/1906.02629)

## Exact Loss Semantics

For a flat output `z`, target `y`, existing per-class weights `w`, and fixed
`epsilon = 0.05`, Candidate Q uses the exact PyTorch semantic:

```python
torch.nn.functional.cross_entropy(
    z,
    y,
    weight=w,
    label_smoothing=0.05,
)
```

The existing hard-CE route stays bit-for-bit equivalent to
`cross_entropy(z, y, weight=w)`. This deliberately preserves A's current
inverse-frequency weighting instead of silently replacing or stacking a second
class-balancing method. The new variant is valid only with the flat classifier;
CORN's conditional BCE and `neutral_gate_polarity`'s normalized log-probability
head are not candidates. R-Drop remains hard-CE-only, so Candidate Q cannot be
combined with R-Drop.

No probability thresholds, calibration, label changes, model parameters,
checkpoint selection, or evaluation rules change. Inference remains `argmax` of
the existing three-class logits.

## Persistence And Isolation

`classification_loss_variant` is a required Q2 TOML field with exactly two
values: `hard_ce` and `weighted_label_smoothing_005`. It is validated when
loading a new configuration, written to `run_manifest.json`, and validated during
strict saved-valid reconstruction. Historical manifests missing the field are
interpreted as `hard_ce`.

The runner passes this value only to classification-loss construction. It does not
alter the model architecture, optimizer, data loader, masks, normalizer, encoding,
prediction code, attachment 3 inference, or valid metric implementation. R-Drop
and non-flat classifier combinations fail before training.

## Tests And Experiment Gate

Before any real run, tests must prove:

- both persisted values parse and unsupported values fail exactly;
- label smoothing is accepted only with `classification_variant="flat"` and
  `dropout_consistency_variant="none"`;
- `hard_ce` remains the current weighted CE, while Candidate Q equals the exact
  weighted PyTorch label-smoothing loss and back-propagates finite gradients;
- fake-archive training records the variant without accessing an inaccessible
  Attachment 2 test sentinel; and
- strict saved-valid reconstruction accepts the persisted Q manifest and rejects
  an unsupported loss value without weakening state-dict loading.

After focused and full unit tests pass, one ignored configuration must preflight to
`train=3395`, `valid=728`, and `attachment3=30` without making an output. It then
trains exactly once into `artifacts/q2-valid-label-smoothing-005/`; a distinct
strict saved-valid report is generated exactly once. The comparison record must
prove that normalized A/Q manifests differ only in
`classification_loss_variant`, and must record clean metrics, per-class F1,
27-scenario mean/worst, 30 attachment-3 predictions, and the explicit
train/valid-only scope.

If clean valid macro-F1 misses `0.6212527658`, retire this exact loss semantics.
Do not tune epsilon, class weights, loss weighting, checkpoints, seed, or combine
it with any retired candidate. Test remains unassessed regardless of outcome.
