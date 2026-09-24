# Q2 Pairwise Hadamard Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one `pairwise_hadamard_residual` Q2 fusion variant and evaluate it exactly once from A using Attachment 2 train/valid only, accepting only clean valid macro-F1 `>= 0.6212527658`.

**Architecture:** Preserve A's masked projections and gated fused state. At each non-padding time slot, add the average of the available pairwise Hadamard products `(text*audio, text*vision, audio*vision)` to that fused state; the residual is exact zero if fewer than two modalities are available. Reuse every existing train/valid, temporal encoder, pooling, head, loss, manifest, and saved-valid path.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers, pytest, local 7-Zip and local BERT assets.

---

### Task 1: Register Only the New Fusion Name

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing variant-contract tests.**

Add `pairwise_hadamard_residual` to the supported config-value parameterization. Add a model construction test equivalent to the existing text-anchor smoke test:

```python
model = MaskAwareTemporalFusion(
    hidden_size=16,
    heads=4,
    layers=1,
    dropout=0.0,
    fusion_variant="pairwise_hadamard_residual",
)
output = model(text=text, audio=audio, vision=vision, masks=masks)
assert output.logits.shape == (2, 3)
assert torch.isfinite(output.logits).all()
assert torch.isfinite(output.score).all()
assert torch.isin(output.logits.argmax(dim=1), torch.tensor([0, 1, 2])).all()
```

Update every exact unsupported-fusion expectation to include the new public value. Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_model.py -q
```

Expected RED: config construction rejects the new name before forward execution; after adding the name, the current gated path may make the smoke assertion pass but does not yet establish the residual behavior.

- [ ] **Step 2: Permit exactly the named value.**

Change only the constant and rejection string:

```python
FUSION_VARIANTS = (
    "gated",
    "mag_lite",
    "mult_lite",
    "late_expert_shared",
    "text_anchor_residual",
    "pairwise_hadamard_residual",
)
```

Do not add settings, defaults, validation branches, or numeric fusion controls.

- [ ] **Step 3: Verify GREEN and commit.**

Run the focused command above, then:

```bash
git add src/e_mosei_audit/q2/config.py tests/test_q2_config.py tests/test_q2_model.py
git commit -m "feat: configure Q2 pairwise Hadamard fusion"
```

### Task 2: Implement the Zero-Parameter Pairwise Residual

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing algebra and mask tests.**

Import a new module-private `_pairwise_hadamard_residual` helper in the model test. Its hand-calculated fixture must cover one available modality, text+audio, all three modalities, and a padding slot:

```python
states = (
    torch.tensor([[[2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]]),
    torch.tensor([[[3.0, 5.0], [3.0, 5.0], [3.0, 5.0], [3.0, 5.0]]]),
    torch.tensor([[[7.0, 11.0], [7.0, 11.0], [7.0, 11.0], [7.0, 11.0]]]),
)
availability = torch.tensor([[[True, False, False], [True, True, False], [True, True, True], [True, True, True]]])
temporal = torch.tensor([[True, True, True, False]])
expected = torch.tensor([[[0.0, 0.0], [6.0, 20.0], [41.0 / 3.0, 119.0 / 3.0], [0.0, 0.0]]])
torch.testing.assert_close(_pairwise_hadamard_residual(states, availability, temporal), expected)
```

Add three public-behavior tests with equal gated/pairwise weights and `dropout=0.0`:

1. A normal all-modality fixture changes logits or score, proving the residual reaches the heads.
2. A text-only-per-row fixture produces exact equality of logits, score, gates, temporal attention and optional public outputs with gated; assert identical state-dict keys and total parameter counts.
3. Partial availability masks with large finite replacements at every unavailable raw text/audio/vision location produce exact equality of logits, score, gates and temporal attention.

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
```

Expected RED: importing the helper fails before implementation; after a temporary helper stub, the all-three-pair hand calculation and normal-availability difference must fail against the unmodified gated path.

- [ ] **Step 2: Add the pure helper.**

After `_projection`, define the no-parameter helper using only already masked states:

```python
def _pairwise_hadamard_residual(
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    availability: torch.Tensor,
    temporal_mask: torch.Tensor,
) -> torch.Tensor:
    pair_masks = (
        temporal_mask & availability[..., 0] & availability[..., 1],
        temporal_mask & availability[..., 0] & availability[..., 2],
        temporal_mask & availability[..., 1] & availability[..., 2],
    )
    pair_products = (
        states[0] * states[1],
        states[0] * states[2],
        states[1] * states[2],
    )
    residual = sum(mask.unsqueeze(-1) * product for mask, product in zip(pair_masks, pair_products, strict=True))
    pair_count = sum(mask.to(dtype=states[0].dtype) for mask in pair_masks)
    return residual / pair_count.clamp_min(1).unsqueeze(-1)
```

The masks make residual values zero for padding and unavailable modalities; `clamp_min(1)` only prevents division by zero and is not a tunable scale.

- [ ] **Step 3: Apply the helper only to K.**

Immediately after the existing `fused = fused.masked_fill(~temporal.unsqueeze(-1), 0.0)` and before the existing temporal encoder call, add:

```python
if self.fusion_variant == "pairwise_hadamard_residual":
    fused = fused + _pairwise_hadamard_residual(states, availability, masks.temporal)
```

Do not modify the gates, temporal masks, pooler, heads, optimizer, losses, state-dict structure, BERT calls, or the other variants.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused model suite, `git diff --check`, then:

```bash
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 pairwise Hadamard residual fusion"
```

### Task 3: Cover Generic Train/Valid Persistence Without Test Access

**Files:**
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Add a fake-archive training boundary test.**

Using `TrainValidPayloadWithInaccessibleTest`, add a `run_q2` test with `fusion_variant="pairwise_hadamard_residual"`. Its exact assertions must include:

```python
assert summary["attachment3_count"] == 30
assert len(predictions) == 30
assert manifest["training"]["fusion_variant"] == "pairwise_hadamard_residual"
assert archive.verify_count == 1
```

The fake Attachment 2 mapping contains only accessible `train`/`valid`; both `payload["test"]` and `payload.get("test")` must raise the existing test-split sentinel assertion.

- [ ] **Step 2: Add strict saved-valid reconstruction coverage.**

Create and save `MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="pairwise_hadamard_residual")`. Use a manifest with that exact fusion name and a `TrainValidPayloadWithInaccessibleTest`. Monkeypatch `q2_runner.MaskAwareTemporalFusion` with a delegating constructor spy and monkeypatch the real class's `load_state_dict` with a delegating spy. After `evaluate_saved_q2_valid`, require:

```python
assert observed_variants == ["pairwise_hadamard_residual"]
assert observed_strict == [True]
assert report["sample_count"] == 3
```

Do not change `run_q2`, `evaluate_saved_q2_valid`, data loaders, manifest writers, `load_aligned_test`, or any production path. The generic runner is expected to pass once Tasks 1-2 are complete.

- [ ] **Step 3: Verify and commit.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
```

Commit only the runner tests:

```bash
git add tests/test_q2_runner.py
git commit -m "test: cover Q2 pairwise valid-only runner"
```

### Task 4: Run and Audit Candidate K Exactly Once

**Files:**
- Create: `q2-pairwise-hadamard-residual.toml` (ignored local configuration)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-pairwise-hadamard-residual/` (ignored)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-pairwise-hadamard-residual.json` (ignored)
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Create the isolated configuration.**

Copy A's resolved fields exactly, set:

```toml
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-pairwise-hadamard-residual"
fusion_variant = "pairwise_hadamard_residual"
classification_variant = "flat"
```

Retain `seed=20260924`, `learning_rate=0.001`, `synthetic_missingness_enabled=false`, `text_adapter_variant="identity"`, 30 epochs, batch size 64, CUDA, and every other A field. Require the target directory to be absent.

- [ ] **Step 2: Preflight and run once.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-pairwise-hadamard-residual.toml --check
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-pairwise-hadamard-residual.toml
```

Require preflight `3395/728/30`. Execute the non-check command exactly once, never load or report Attachment 2 test, and never rerun to chase the result.

- [ ] **Step 3: Audit, decide, and document.**

Require a nonempty model and manifest, 728 valid-report support, 27 scenario rows, 30 Attachment 3 predictions, and normalized A/K manifest difference only `fusion_variant: gated -> pairwise_hadamard_residual`. Write the comparison JSON from actual artifacts. Accept only macro-F1 `>=0.6212527658`; record a Test row as `未评估`, update the portfolio, run full pytest and `git diff --check`, then commit the documentation.
