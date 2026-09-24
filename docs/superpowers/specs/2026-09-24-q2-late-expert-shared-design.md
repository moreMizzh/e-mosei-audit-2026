# Q2 Shared Late-Expert Fusion Design

## Decision

Candidate G is a single, valid-only treatment derived independently from A. Its sole configuration change is `fusion_variant="late_expert_shared"`; `text_adapter_variant="identity"` and every other input, training, checkpoint-selection and output setting stays identical to A. This design tests whether the baseline's frame-level multimodal fusion is harmful, without testing a larger model at the same time.

Three alternatives were considered. Fixed averaging has no learned reliability signal and cannot represent the existing missingness condition. Private modality encoders and heads introduce roughly three times the temporal capacity, which confounds fusion placement with model size on 3,395 training examples. The selected shared-expert design uses the existing temporal encoder, pool attention, classifier, regressor and gate with no new trainable parameter or width.

## Data Flow

After the unchanged projections and source-value masking, let `state_m` be the projected text, audio, or vision state and let `p_m = masks.temporal & availability_m`.

1. For each modality `m`, only rows for which `p_m.any(dim=1)` are passed to the existing `temporal_encoder`, with `src_key_padding_mask=~p_m`. Inactive rows never enter a Transformer with all padding. The encoded active rows are masked again, and the existing `pool_attention` produces a modality representation `h_m[B,H]` and modality attention `alpha_m[B,T]`; inactive rows are all zero.
2. Define `coverage_m = sum_T(p_m) / sum_T(masks.temporal)`. The existing classifier and regressor receive `concat(h_m, one_hot(m) * coverage_m)`, which preserves their `[H+3]` input contract while stating which modality produced the expert prediction. They emit expert classification logits `ell_m[B,3]` and bounded scores `s_m[B]`.
3. The existing gate receives `concat(h_text, h_audio, h_vision, coverage_text, coverage_audio, coverage_vision)` with shape `[B,3H+3]`. Its logits are masked to `-inf` for globally inactive experts before softmax, yielding `w[B,3]`. Each sample is known to retain at least one active modality, so the masked softmax is well-defined.
4. Final predictions are `sum_m(w_m * ell_m)` and `sum_m(w_m * s_m)`. The latter remains in `[-3,3]` because it is a convex combination of bounded scores. No new loss, temperature, coefficient, augmentation, checkpoint rule, or calibration rule is introduced.

For compatibility, `Q2Output.gates` remains `[B,T,3]` and records the applied expert weight `w_m` only at positions where `p_m` is true; it is zero at unavailable positions. `Q2Output.temporal_attention` remains `[B,T]` as `sum_m(w_m * alpha_m)`. A new optional `expert_weights[B,3]` exposes the scalar late-fusion reliability weights for this variant; it is `None` for existing variants.

## Missingness and Determinism

The existing pre-projection zeroing remains mandatory. Both the per-expert temporal mask and the expert availability mask use `p_m`, so changing raw values at unavailable positions must not change logits, score, applied gates, temporal attention, or expert weights. An entirely missing expert contributes a zero representation, zero expert prediction, zero attention and exactly zero reliability weight.

`late_expert_shared` does not instantiate expert-specific modules. Consequently its common module state and CPU RNG sequence are byte-identical to `gated` under the same seed; this must be protected by a regression test. The old `gated`, `mag_lite`, `mult_lite` and text-adapter construction paths retain their behavior. Legacy manifests without `fusion_variant` still reconstruct `gated`.

## Training Protocol and Acceptance

The one real run uses Attachment 2 `train=3395` and `valid=728` only, with seed `20260924`, epochs `30`, batch size `64`, learning rate `0.001`, weight decay `0.0001`, hidden size `128`, four heads, two layers, dropout `0.1`, regression loss weight `0.5`, polarity-consistency weight `0.0`, class-weight exponent `1.0`, CUDA, local BERT and synthetic missingness disabled. Attachment 2 `test` is not loaded, indexed, trained on, evaluated, selected on, or reported.

The real output directory is new and ignored. It must include a nonempty model and manifest, a valid classification support total of `728`, exactly `27` validation scenario rows and exactly `30` unlabeled Attachment 3 predictions. The comparison normalizes A's legacy manifest with `polarity_consistency_loss_weight=0.0`, `fusion_variant="gated"`, and `text_adapter_variant="identity"`; only `fusion_variant` may differ.

Accept the single treatment only when all conditions hold: clean valid macro-F1 is at least `0.6212527658`; clean valid MAE is at most `0.6446957182884216`; and 27-scenario macro-F1 mean/worst are at least `0.5937955228962313` / `0.5363436545162387`. Otherwise permanently reject this exact shared late-expert structure without modifying encoder depth, head count, gate, coverage encoding, expert heads, initialization, checkpoint or threshold.

## Verification

Before the real run, unit and fake-archive tests must cover config acceptance, finite outputs and bounds, exact isolation of unavailable raw values, fully missing experts, partial masks, nonzero finite gradients through the shared encoder/gate/heads, unchanged gated initialization/RNG, persisted manifest value and strict saved-valid checkpoint reload with an inaccessible test sentinel. The full suite, `train-q2 --check`, one real training run, artifact-count checks, comparison JSON validation and README Val/Test table update follow only after the implementation passes.
