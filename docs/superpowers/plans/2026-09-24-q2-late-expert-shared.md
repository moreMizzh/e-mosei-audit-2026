# Q2 Shared Late-Expert Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one fixed, parameter-shared, availability-masked late-expert Q2 fusion variant and evaluate it once against A using Attachment 2 train/valid only.

**Architecture:** `late_expert_shared` reuses the baseline projections, temporal encoder, pooling attention, classifier, regressor and gate. Each projected modality is encoded independently through the shared temporal path, then modality-tagged expert predictions are combined only at the final logit/score layer by a masked reliability gate. The implementation introduces no trainable parameter, width, loss, augmentation, checkpoint or calibration knob beyond the existing `fusion_variant`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, local BERT cache and explicit 7-Zip.

---

### Task 1: Accept and Persist the Fixed Fusion Name

**Files:**

- Modify: `src/e_mosei_audit/q2/config.py:14,190-195`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_model.py:394-396`
- Modify: `tests/test_q2_runner.py:813-854`

- [ ] **Step 1: Write failing configuration tests.**

Extend the existing supported-fusion parametrization to include `"late_expert_shared"`, and add this direct constructor assertion:

```python
def test_mask_aware_fusion_accepts_late_expert_shared_variant() -> None:
    model = MaskAwareTemporalFusion(fusion_variant="late_expert_shared")
    assert model.fusion_variant == "late_expert_shared"
```

Update every rejected-fusion error expectation to the exact new contract:

```python
match="fusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared"
```

- [ ] **Step 2: Run the focused tests and confirm RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest \\
  tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py -q
```

Expected: the new constructor/config case fails because `late_expert_shared` is absent from `FUSION_VARIANTS`; legacy rejected-variant tests still pass before their expected message is changed.

- [ ] **Step 3: Implement only the enum extension.**

Change the configuration constant and its exact validation message to:

```python
FUSION_VARIANTS = ("gated", "mag_lite", "mult_lite", "late_expert_shared")

def validate_fusion_variant(value: object) -> str:
    if not isinstance(value, str) or value not in FUSION_VARIANTS:
        raise ValueError(
            "fusion_variant must be one of: gated, mag_lite, mult_lite, late_expert_shared"
        )
    return value
```

Do not add a TOML field, a hidden fallback or an alternate name. The existing runner already passes this value to the model, writes it into `run_manifest.json`, and validates it while rebuilding a saved checkpoint.

- [ ] **Step 4: Verify GREEN and commit the narrow contract.**

Run the Step 2 command and `git diff --check`. Commit only the configuration and test changes:

```bash
git add src/e_mosei_audit/q2/config.py tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py
git commit -m "feat: configure Q2 shared late expert fusion"
```

### Task 2: Implement the Shared Late-Expert Forward Path by TDD

**Files:**

- Modify: `src/e_mosei_audit/q2/model.py:27-31,135-215`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing behavioral tests.**

Add a finite-output and inspectability test for an all-observed batch:

```python
def test_late_expert_shared_returns_finite_bounded_predictions_and_expert_weights() -> None:
    model = MaskAwareTemporalFusion(
        hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="late_expert_shared"
    )
    output = model(
        text=torch.randn(2, 50, 768), audio=torch.randn(2, 50, 74),
        vision=torch.randn(2, 50, 35), masks=example_masks(),
    )
    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert output.gates.shape == (2, 50, 3)
    assert output.temporal_attention.shape == (2, 50)
    assert output.expert_weights is not None
    assert output.expert_weights.shape == (2, 3)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.score).all()
    assert torch.isfinite(output.gates).all()
    assert torch.isfinite(output.temporal_attention).all()
    assert torch.isfinite(output.expert_weights).all()
    torch.testing.assert_close(output.expert_weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.score <= 3)
    assert torch.all(output.score >= -3)
```

Add a full/partial missingness test with text fully available, audio unavailable for the second sample, vision unavailable for the first sample, and additional unavailable positions in the active rows. Run the same evaluation once with raw values at every unavailable text/audio/vision position replaced by `1_000_000.0`. Require exact equality of `logits`, `score`, `gates`, `temporal_attention` and `expert_weights`; require zero `expert_weights[1, 1]`, zero `expert_weights[0, 2]`, and zero applied gates at all unavailable positions.

Wrap `model.temporal_encoder` in a test-only recording module that stores every `src_key_padding_mask`. For the preceding masks, require exactly three calls per forward: the text call has both rows and mask `~text_available`, the audio call has only its active first row and mask `~audio_available[0:1]`, and the vision call has only its active second row and mask `~vision_available[1:2]`. Assert every recorded row contains at least one unmasked temporal position.

Add a gradient test on an all-observed batch with loss `output.logits.square().mean() + output.score.square().mean()`. Require finite, nonzero gradients for every parameter in `temporal_encoder`, `pool_attention`, `gate`, `classifier` and `regressor`.

Finally add the initialization regression:

```python
torch.manual_seed(53)
gated = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
gated_next = torch.rand(4)
torch.manual_seed(53)
late = MaskAwareTemporalFusion(
    hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="late_expert_shared"
)
late_next = torch.rand(4)
assert gated.state_dict().keys() == late.state_dict().keys()
assert all(torch.equal(gated.state_dict()[name], late.state_dict()[name]) for name in gated.state_dict())
assert torch.equal(gated_next, late_next)
```

- [ ] **Step 2: Run the model tests and confirm RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
```

Expected: new output tests fail because `Q2Output` has no `expert_weights` and `MaskAwareTemporalFusion.forward` still applies the frame-level gate before its temporal encoder.

- [ ] **Step 3: Add the optional inspection field and the late path.**

Extend `Q2Output` without altering existing callers:

```python
@dataclass(frozen=True)
class Q2Output:
    logits: torch.Tensor
    score: torch.Tensor
    gates: torch.Tensor
    temporal_attention: torch.Tensor
    expert_weights: torch.Tensor | None = None
```

Immediately after computing `states`, `availability` and the existing nonempty `temporal` check, dispatch this variant before MAG, MulT or frame-level gating:

```python
if self.fusion_variant == "late_expert_shared":
    return self._forward_late_expert_shared(states, availability, masks.temporal)
```

Implement `_forward_late_expert_shared` with this fixed data flow:

```python
expert_masks = tuple(temporal_mask & availability[..., index] for index in range(3))
representations, attentions = zip(
    *(self._encode_late_expert(state, expert_mask) for state, expert_mask in zip(states, expert_masks, strict=True)),
    strict=True,
)
representation = torch.stack(representations, dim=1)
attention = torch.stack(attentions, dim=1)
temporal_count = temporal_mask.sum(dim=1, keepdim=True).to(dtype=representation.dtype)
coverage = torch.stack(expert_masks, dim=-1).sum(dim=1).to(dtype=representation.dtype) / temporal_count
modality_features = coverage.unsqueeze(-1) * torch.eye(3, device=representation.device, dtype=representation.dtype)
expert_features = torch.cat((representation, modality_features), dim=-1)
expert_active = torch.stack(expert_masks, dim=-1).any(dim=1)
expert_logits = self.classifier(expert_features).masked_fill(~expert_active.unsqueeze(-1), 0.0)
expert_scores = 3.0 * torch.tanh(self.regressor(expert_features).squeeze(-1)).masked_fill(~expert_active, 0.0)
gate_features = torch.cat((representation.flatten(start_dim=1), coverage), dim=1)
gate_logits = self.gate(gate_features).masked_fill(~expert_active, float("-inf"))
expert_weights = torch.softmax(gate_logits, dim=1)
logits = torch.sum(expert_weights.unsqueeze(-1) * expert_logits, dim=1)
score = torch.sum(expert_weights * expert_scores, dim=1)
applied_gates = expert_weights.unsqueeze(1) * torch.stack(expert_masks, dim=-1).to(dtype=representation.dtype)
temporal_attention = torch.sum(expert_weights.unsqueeze(-1) * attention, dim=1)
return Q2Output(logits, score, applied_gates, temporal_attention, expert_weights)
```

Implement `_encode_late_expert` as follows. Start with zero `[B,H]` representations and zero `[B,T]` attention; select only `expert_mask.any(dim=1)` rows; zero masked state positions before calling the existing `self.temporal_encoder` with `src_key_padding_mask=~active_mask`; zero masked encoded positions; attention-pool; and `index_copy` the results back. Return all zeros when no row is active. This prevents all-padding Transformer calls and makes unavailable raw values observationally irrelevant.

Do not create a new encoder, pool, head, gate, loss, parameter or configuration field. Do not alter construction order: because the variant only changes `forward`, the gated initialization and CPU RNG stream remain byte-identical.

- [ ] **Step 4: Run focused and full model tests, then commit.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_model.py -q
git diff --check
```

Expected: all listed tests pass. Commit only the model and its tests:

```bash
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 shared late expert fusion"
```

### Task 3: Prove Runner Persistence and Valid-Only Checkpoint Recovery

**Files:**

- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write a failing fake-archive training test.**

Duplicate the existing MulT runner fixture, set only:

```python
config = replace(
    runner_config(tmp_path),
    output_dir=tmp_path / "q2-late-expert-output",
    fusion_variant="late_expert_shared",
    text_adapter_variant="identity",
)
```

Make the fixture's `test` member an inaccessible sentinel rather than a split. Run `run_q2`, then assert the manifest writes `"fusion_variant": "late_expert_shared"`, `"text_adapter_variant": "identity"`, the output has 30 Attachment 3 predictions and the fake archive's test sentinel was never requested.

Add a saved-valid test that builds `MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="late_expert_shared")`, saves its state dict and a manifest naming the same variant, then calls `evaluate_saved_q2_valid`. Require `sample_count == 3` and a single archive verification while the fake `test` member remains inaccessible.

- [ ] **Step 2: Run the runner tests and confirm RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
```

Expected before Tasks 1 and 2 are complete: variant validation or strict checkpoint reconstruction fails. Expected after Tasks 1 and 2 are complete: both tests pass without any runner source edit, proving its existing train/valid-only loader and manifest plumbing remain sufficient.

- [ ] **Step 3: Run all runner checks and commit the boundary coverage.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
```

Expected: the full suite passes, and any real-data opt-in tests remain skipped unless their explicit environment variable is supplied. Commit the new runner tests:

```bash
git add tests/test_q2_runner.py
git commit -m "test: cover Q2 shared late expert runner"
```

### Task 4: Run One Pre-Registered Valid-Only Treatment and Record It

**Files:**

- Modify (ignored): `q2.toml`
- Create (ignored): `artifacts/q2-valid-late-expert-shared/`
- Create (ignored): `artifacts/q2-valid-comparison-no-train-missingness-late-expert-shared.json`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Configure the one permitted real run.**

In ignored `q2.toml`, retain every A setting and set exactly:

```toml
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-late-expert-shared"
fusion_variant = "late_expert_shared"
text_adapter_variant = "identity"
```

Keep archive, seven-zip, BERT path, seed, epochs, batch size, optimizer values, hidden size, heads, layers, dropout, regression loss, consistency loss, class weighting, CUDA and `synthetic_missingness_enabled=false` exactly as in A. Confirm the output directory does not exist.

- [ ] **Step 2: Verify inputs, then train exactly once.**

Run the preflight and require exactly `train_count=3395`, `valid_count=728`, `attachment3_count=30`:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml --check
```

Then run this command once and do not rerun it after observing its metrics:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml
```

The runner must continue to use `load_aligned_train_valid`; never invoke a test split loader, `evaluate-q2-valid` on Attachment 2 test, multi-seed execution or a post-hoc score/threshold search.

- [ ] **Step 3: Audit the run and decide without tuning.**

Require nonempty `model.pt` and `run_manifest.json`; four clean metrics; three-class valid support totaling 728; 27 scenario rows; and 30 Attachment 3 prediction rows. Normalize the control manifest with `polarity_consistency_loss_weight=0.0`, `fusion_variant="gated"`, `text_adapter_variant="identity"`; compare archive/tool/model references, all fixed training fields and each normalizer array exactly. The sole diff must be `fusion_variant`.

Create the ignored comparison JSON with the existing schema plus the exact valid-only scope, A/candidate clean metrics, candidate-minus-control deltas, 27-row scenario mean/worst and their deltas, three-class F1 values, `728/27/30` counts, normalized manifest diff and one final decision. Accept only if macro-F1 is at least `0.6212527658`, MAE is at most `0.6446957182884216`, and scenario mean/worst F1 are at least `0.5937955228962313` / `0.5363436545162387`. On any failure, permanently reject the precise `late_expert_shared` architecture: no head, depth, gate, coverage, initialization, calibration or checkpoint tuning is allowed.

- [ ] **Step 4: Update the score ledger and commit the result.**

Append one `late_expert_shared` row to both README tables. Take Val values exclusively from the new `metrics.json`, with six decimals and the actual manifest best epoch. Use `-` for every Test metric and `未评估` status. Add a portfolio stage record with the exact decision and comparison JSON path; make the next route conditional on the documented portfolio rather than altering G after the fact.

Validate the comparison JSON, run the full test suite and check formatting:

```bash
/home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m json.tool /home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-late-expert-shared.json >/dev/null
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
```

Commit only the tracked documentation result:

```bash
git add README.md docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
git commit -m "docs: record Q2 shared late expert result"
```
