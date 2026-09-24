# Q2 Availability-Conditioned Temporal Pooling Candidate Design

## Goal And Boundary

Candidate N is independently derived from the current sequential control A, artifacts/q2-valid-no-train-missingness. Its single normalized training difference is:

    temporal_pooling_variant: "attention" -> "attention_availability"

It keeps temporal_position_variant="none", fusion_variant="gated", seed 20260924, 30 epochs, batch size 64, learning rate 0.001, weight decay 0.0001, frozen local BERT, all existing losses and clean-valid checkpoint selection, and synthetic_missingness_enabled=false. It will run exactly once on Attachment 2 official train=3395 and valid=728. It is accepted only at clean valid macro-F1 >= 0.6212527658.

Attachment 2 test must not be read, indexed, validated, trained on, used for early stopping, selected, evaluated, or reported. Attachment 3 remains exactly 30 unlabeled inference rows and cannot influence selection. Candidate M's sinusoidal position formula is retired after its clean macro-F1 0.5819145435; do not change its formula, base, placement, scale, positional type, checkpoint, or seed.

## Rationale

A's shared temporal encoder uses gated fused states, but its scalar readout scores only encoded content:

    attention_logits[t] = pool_attention(encoded[t])

At each time slot, the three modality availability bits already control fusion. They reappear after pooling only as an utterance-level availability fraction. Thus the readout cannot directly learn whether a locally encoded state arose from one, two, or three observed modalities. Candidate M established that forcing global order information is harmful on this 50-slot representation; N instead tests a local reliability signal already defined by the existing mask contract.

The change is not residual fusion, multiplicative interaction, MAG, MulT, late experts, CORN, text adapter, loss weighting, learning-rate tuning, or synthetic missingness. It targets the remaining pool-readout blind point with three trainable scalars.

## Exact Model Change

Let h[b,t] be the existing shared temporal encoder output, a[b,t] be the existing boolean availability vector in {0,1}^3, and temporal[b,t] be its existing non-padding and at-least-one-modality mask. A uses:

    l_A[b,t] = w_pool^T h[b,t] + c_pool
    alpha[b] = softmax(mask(l_A[b], temporal[b]))
    pooled[b] = sum_t alpha[b,t] * h[b,t]

N adds exactly a bias-free linear map r in R^(3 x 1) and replaces only the attention logits:

    l_N[b,t] = l_A[b,t] + r^T float(a[b,t])
    alpha_N[b] = softmax(mask(l_N[b], temporal[b]))
    pooled_N[b] = sum_t alpha_N[b,t] * h[b,t]

r starts at exactly zero, so the candidate's epoch-0 forward is exactly A for identical base parameters. The map uses availability booleans, never raw text/audio/vision values, and runs before the existing temporal mask is changed to negative infinity. The existing softmax, pooled representation, availability fraction, classifier, regressor, loss, optimizer, epoch loop, and checkpoint comparison stay unchanged.

temporal_pooling_variant has exactly two values: "attention" and "attention_availability". The latter creates pool_availability_bias = nn.Linear(3, 1, bias=False) with all-zero weights. Its construction must preserve successor RNG state. It adds exactly 3 parameters and one state-dict key. Historical manifests that lack the new field default to "attention"; strict loading therefore preserves the meaning of both historical A checkpoints and N checkpoints.

## Required Invariants

- At zero initialization, candidate N has identical base parameters, successor RNG state, logits, score, gates, temporal attention, and output to a matched A model. It differs only by three zero weights and their state-dict key.
- Availability-bias weights receive finite, nonzero gradients in a varied-availability batch. They must remain unavailable to all-invalid or padded slots through the existing temporal negative-infinity mask.
- Altering unavailable raw modality values to large finite values cannot change outputs. The extra path sees only masks.
- Appended fully unavailable padding has zero temporal attention and must not affect valid behavior apart from bounded shape-dependent floating reductions already established for CPU linear projections.
- Fresh configs, run manifests, saved-valid reconstruction, and invalid-manifest handling must persist the exact pooling variant. Legacy manifest absence must map only to "attention", never silently accept an invalid present value.
- Fake-archive training and saved-valid tests must make any Attachment 2 test access fail.

## Evaluation And Decision

Before real execution: write and observe failing tests, pass focused suites and the full suite, run train-q2 --check with exact 3395/728/30, and confirm artifacts/q2-valid-attention-availability does not exist. The one run must emit a model, manifest, 728-row valid classification report, 27 missingness scenarios, 30 Attachment 3 predictions, and a strict saved-valid recomputation.

Write a comparison JSON against A, including the 3-parameter change, clean metrics, class F1 values, scenario mean/worst macro-F1, normalizer equality, counts, manifest-normalized difference, and explicit valid-only scope. Add actual values to the README Val and Test tables; Test must remain unassessed. If N misses the threshold, retire exactly this zero-initialized availability-bias pooling route without sweeping initialization, scale, bias, activation, availability features, loss, optimizer, checkpoint, or seed.

