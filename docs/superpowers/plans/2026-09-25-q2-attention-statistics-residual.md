# Q2 Attention Statistics Residual Candidate V Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans task by task. Steps use checkbox syntax.

**Goal:** Implement and run exactly one A-derived, valid-only Candidate V whose sole normalized change is an attention-weighted statistics residual in temporal pooling.

**Architecture:** Add `attention_statistics_residual` to the persisted pooling enum. Existing attention weights and weighted mean stay intact; a zero-initialized `[H]` scale multiplies the attention-weighted standard deviation. Existing generic runner persistence and strict replay must be proved by tests, not changed speculatively.

**Files:** `src/e_mosei_audit/q2/config.py`, `src/e_mosei_audit/q2/model.py`, `tests/test_q2_config.py`, `tests/test_q2_model.py`, `tests/test_q2_runner.py`; after the real result, README and the Q2 portfolio.

### Task 1: Cross-Layer RED Tests

- [ ] Add `"attention_statistics_residual"` to the supported pooling parametrization in `tests/test_q2_config.py`. Update the exact unknown-value error to `temporal_pooling_variant must be one of: attention, attention_availability, attention_statistics_residual, masked_mean`. Keep the complete TOML fixture field `temporal_residual_variant = "none"`.

- [ ] Before production edits, add model tests at `H=16`, `heads=4`, `layers=1`, `dropout=0.0`. Same-seed ordinary attention versus V must establish the only new key `pool_statistics_scale`, its exact zero `[16]` tensor, parameter delta 16, successor Torch RNG equality, and exact public output equality at zero scale. Use `availability_pooling_masks()` and `RecordingEncoder` with a nonzero scale to prove unavailable raw values, an internal all-unavailable slot, and appended fully unavailable `1_000_000.0` padding leave valid outputs unchanged and invalid attention at zero. Backpropagate `logits.square().sum() + score.square().sum()` and require a finite nonzero scale gradient. Same-seed zero-scale ordinary and V `late_expert_shared` models must have exact public output equality.

- [ ] Add fake-runner tests with `replace(runner_config(tmp_path), output_dir=tmp_path / "q2-attention-statistics-output", temporal_pooling_variant="attention_statistics_residual")`. `check_q2` and `run_q2` use `runner_archive_with_inaccessible_test()`, record the model constructor value, preserve check non-output behavior, produce the `3/3/30` fixture counts, persist the variant in manifest, and cannot access Test. A saved `H=16` V checkpoint/manifest strict replay observes the same variant and `load_state_dict(..., strict=True)` with a Test-inaccessible archive.

- [ ] Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py -q`. Expected RED: config rejects V and `pool_statistics_scale` is absent. Do not write production code before that failure.

### Task 2: Persist the Pooling Value

- [ ] In `config.py`, set the exact tuple:

```python
TEMPORAL_POOLING_VARIANTS = (
    "attention",
    "attention_availability",
    "attention_statistics_residual",
    "masked_mean",
)
```

Make `validate_temporal_pooling_variant` emit the exact Task 1 message. Do not add a field, silent default, runner branch, or compatibility rule: the required existing field, manifest serializer, and legacy absent-field fallback are already the public contract.

- [ ] Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q`, then commit config and config/runner test changes as `feat: configure Q2 attention statistics pooling`.

### Task 3: Implement the Frozen Readout

- [ ] In `model.py`, after `pool_attention`, create the sole optional parameter with the existing RNG-preserving style:

```python
if self.temporal_pooling_variant == "attention_statistics_residual":
    rng_state = torch.get_rng_state()
    try:
        self.pool_statistics_scale = nn.Parameter(torch.zeros(hidden_size))
    finally:
        torch.set_rng_state(rng_state)
```

- [ ] Add and call this helper immediately after the current attention-weighted mean in both the ordinary path and `_encode_late_expert`:

```python
def _apply_attention_statistics_residual(self, pooled, encoded, temporal_attention):
    if self.temporal_pooling_variant != "attention_statistics_residual":
        return pooled
    centered = encoded - pooled.unsqueeze(1)
    variance = torch.sum(temporal_attention.unsqueeze(-1) * centered.square(), dim=1)
    statistics = torch.sqrt(variance + torch.finfo(encoded.dtype).eps)
    return pooled + self.pool_statistics_scale * statistics
```

Do not alter masks, attention logits, availability fraction, gates, Transformer, BERT, loss, optimizer, or checkpoint selection.

- [ ] Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py -q`; run `git diff --check`; commit model and model tests as `feat: add Q2 attention statistics pooling`.

### Task 4: Contract Review

- [ ] Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`, `git diff --check main...HEAD`, and `git status --short --branch`.

- [ ] Conduct specification review for the one enum, only `[128]` parameter, exact zero/RNG/output equivalence, both attention paths, masking/padding safety, strict replay, Test sentinels, and A-only configuration. Follow with independent code-quality review of numerical stability, state compatibility, and test strength. Fix all findings before real execution.

### Task 5: One V Run and Decision

- [ ] Create ignored `/home/administrator/MyItem/E/q2-attention-statistics-residual.toml`, preserving A values including `temporal_residual_variant = "none"`, with only `output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-attention-statistics-residual"` and `temporal_pooling_variant = "attention_statistics_residual"` changed. Require the TOML and output path absent; run exactly one `train-q2 --check` and accept only `3395/728/30` with no output. Never invoke a Test command.

- [ ] Run training once and `evaluate-q2-valid` once. Require saved/replayed clean metrics and confusion matrix equality, support 728, 27 scenario rows, 30 Attachment 3 predictions, nonempty model, and sole normalized diff `temporal_pooling_variant: attention -> attention_statistics_residual`.

- [ ] Write ignored comparison JSON with valid-only scope, clean/per-class/scenario deltas, `728/27/30`, strict equality, finite/nonzero `[128]` scale evidence, model SHA-256, and the sole normalized diff. Add paired README Val/Test rows, keeping every V Test value `-` and status `未评估`; append portfolio decision. Below `0.6212527658`, retire the exact statistic/epsilon/initialization/scale/optimizer/checkpoint/seed without retry, sweep, or combination; otherwise promote the frozen V model. Commit docs as `docs: record Q2 attention statistics residual result`.

## Self-Review

The plan covers the frozen design, test-first evidence, generic persistence/replay verification, one preflight, one valid-only run, one strict reconstruction, and the irreversible acceptance decision. The sole new public value is `attention_statistics_residual`; the sole added state is `pool_statistics_scale[hidden_size]`.
