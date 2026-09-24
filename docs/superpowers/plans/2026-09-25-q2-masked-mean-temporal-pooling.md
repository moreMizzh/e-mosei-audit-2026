# Q2 Masked-Mean Temporal-Pooling Implementation Plan

**Goal:** Add the persisted no-parameter `masked_mean` temporal-pooling option
and evaluate exactly one A-derived valid-only Candidate S run.

**Architecture:** Extend only the existing temporal-pooling enum. The model
retains all default parameters and uses the existing post-fusion temporal mask
as normalized uniform weights under `masked_mean`; attention pooling and
availability-conditioned pooling retain their current paths. The runner's
existing manifest/replay wiring records and restores the variant.

**Boundary:** No Attachment 2 Test load, prediction, training decision, early
stopping decision, model selection decision, parameter search, seed change, or
combination with a retired candidate is permitted.

## Task 1: Persisted Pooling Contract And Model Semantics

- [ ] Add `masked_mean` to the accepted temporal pooling values and update
  exact invalid-value expectations in config, model, runner, and CLI tests.
- [ ] Write RED tests before production changes for parser acceptance, uniform
  masked weights, ignored unavailable values and padding, state-dict/RNG
  identity with `attention`, and zero `pool_attention` gradient.
- [ ] Implement the smallest branch that normalizes the Boolean temporal mask
  after encoding. Apply the same pooling rule to `_encode_late_expert`; retain
  all existing branches exactly.
- [ ] Verify focused model/config tests and commit the implementation.

## Task 2: Runner Persistence And Strict Valid Replay

- [ ] Write RED fake-archive tests proving `run_q2` and `check_q2` accept a
  manually constructed `Q2Config(..., temporal_pooling_variant="masked_mean")`
  without Test access; require persisted manifest evidence.
- [ ] Write RED strict saved-valid tests proving a saved masked-mean model is
  reconstructed with that value and an unsupported saved value fails before
  archive access. Preserve the absent-field historical fallback to `attention`.
- [ ] Make only the production changes genuinely required after Task 1 (the
  generic current runner wiring may need none), then run focused and full
  suites plus `git diff --check` and commit.

## Task 3: Single Valid-Only Experiment And Frozen Record

- [ ] Create an ignored A-normalized `q2-masked-mean.toml`, changing only
  `output_dir` and `temporal_pooling_variant = "masked_mean"`.
- [ ] Run one `--check` preflight; require `3395/728/30` and no output.
- [ ] Train exactly once, rebuild saved valid exactly once, and compare against
  A's normalized manifest. Never invoke a Test command.
- [ ] Record actual Val metrics in the paired README table, retain an all-dash
  Test row, document the result in the exploration portfolio, and freeze or
  promote Candidate S strictly from the stated macro-F1 threshold.
