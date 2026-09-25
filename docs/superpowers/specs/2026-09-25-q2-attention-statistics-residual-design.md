# Q2 Attention Statistics Residual Candidate V Design

## Purpose

Find one additional, independently pre-registered Q2 candidate without
accessing Attachment 2 `test`. Candidate A remains the sequential control:
its fixed-seed clean valid macro-F1 is `0.6165677805783404`; the acceptance
gate is `0.6212527658`. Candidate V changes exactly one normalized training
semantic from A and does not continue, tune, or combine any retired candidate
B-U.

## Options Considered

1. **Selected: attention statistics residual.** Keep A's learned temporal
   attention and weighted mean, then make its per-hidden-dimension weighted
   standard deviation available through a zero-initialized residual. It adds
   128 parameters, starts as the exact A function, and keeps all data and
   masking interfaces unchanged. Attentive statistics pooling is a precedent
   for combining attention-weighted mean and standard deviation in sequence
   readout, but this candidate is only a small, fixed hypothesis on the
   present 50-slot representation. [Okabe et al.](https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html)
2. **Deferred: clipped relative-attention bias.** A zero-initialized, head-wise
   relative-distance bias is distinct from retired absolute sinusoidal
   positions, but it changes Transformer attention-score internals and needs a
   custom encoder layer. It remains a later independent hypothesis, not a
   fallback or combination for V. [Shaw et al.](https://aclanthology.org/N18-2074/)
3. **Deferred: training-time logit-adjusted CE.** It would replace A's
   inverse-frequency weighted hard CE using train-only class priors, but the
   present class weighting already targets the same imbalance and it changes
   the optimization target rather than representation. It is a later,
   separately pre-registered loss hypothesis. [Menon et al.](https://openreview.net/forum?id=VvRkyPMo6E)

V is not the retired `masked_mean` replacement (S), availability-logit bias
(N), availability embedding (T), local depthwise convolution (U), or any
combination thereof.

## Frozen Candidate V

Add one persisted pooling value:

```toml
temporal_pooling_variant = "attention_statistics_residual"
```

The real V configuration derives from A and retains `fusion_variant="gated"`,
`temporal_position_variant="none"`, `temporal_context_variant="none"`,
`temporal_residual_variant="none"`, flat weighted hard CE, frozen BERT final
layer, `synthetic_missingness_enabled=false`, seed `20260924`, 30 epochs,
batch size 64, learning rate `0.001`, and weight decay `0.0001`. Its
normalized manifest may differ from A only in `temporal_pooling_variant`.

After the current attention logits have been masked by `temporal` and
softmaxed into `alpha`, and after the existing weighted mean
`p = sum_t alpha_t * h_t`, V creates exactly one additional state key:

```python
self.pool_statistics_scale = nn.Parameter(torch.zeros(hidden_size))
```

Its construction restores the Torch RNG state. No projection, normalization,
bias, gate, activation, loss, or auxiliary task is added. For finite encoded
states, it computes the attention-weighted population standard deviation and
the residual only as:

```python
centered = encoded - pooled.unsqueeze(1)
variance = torch.sum(temporal_attention.unsqueeze(-1) * centered.square(), dim=1)
statistics = torch.sqrt(variance + torch.finfo(encoded.dtype).eps)
pooled = pooled + self.pool_statistics_scale * statistics
```

At initialization, `pool_statistics_scale` is exactly zero, so common state,
successor Torch RNG, and public predictions are exactly equal to A under the
same seed. Invalid positions already have zero temporal attention; they cannot
contribute to the weighted variance. Inputs with no available modality are
removed before the Transformer by the existing masks, and appending fully
unavailable padding must not change valid predictions. The statistic is finite
because every sample already has at least one valid temporal slot and epsilon
is dtype-local.

The variant applies to any existing temporal-attention path, including the
shared late-expert helper, so it does not introduce an unvalidated
fusion-specific failure. The only real V run, however, is the A-derived
gated configuration above.

## Evidence And Safety Contracts

Before the real run, tests must establish all of the following.

- Fresh configs accept exactly the existing pooling values plus
  `attention_statistics_residual`; unknown values continue to fail with a
  complete, deterministic enum message. Old manifests that omit the field
  still reconstruct `attention`.
- Relative to ordinary `attention`, V adds only
  `pool_statistics_scale[H]`, initially all zeros. Common state, public
  outputs, and successor Torch RNG values are exact matches.
- With a nonzero test scale, unavailable raw feature values, internal
  all-unavailable slots, and appended `1_000_000.0` padding cannot alter valid
  outputs. Invalid slots retain zero temporal attention and cannot enter the
  weighted statistics.
- A varied valid batch gives the scale a finite, nonzero gradient. Its
  zero-initialized state does not require a changed optimizer, seed, or
  checkpoint policy.
- `check_q2`, fake `run_q2`, manifest storage, and strict saved-valid replay
  preserve the pooling value while fake archive tests make Attachment 2 Test
  access fail. Strict state loading remains enabled.

The ignored V TOML must preflight as `train=3395`, `valid=728`, and
`attachment3=30` without creating output. Only after the full suite, one
preflight, and artifact absence are verified may it train once and reconstruct
valid once. Record clean metrics, class F1, 27-scenario mean/worst, 728/27/30
counts, strict-replay equality, model evidence, and the sole normalized
manifest difference. Accept only clean valid macro-F1 `>= 0.6212527658`;
otherwise permanently retire this exact statistic, epsilon placement,
initialization, scale, optimizer, checkpoint, and seed without retry, sweep,
or combination. Attachment 2 Test is never read, evaluated, selected on, or
reported for V.
