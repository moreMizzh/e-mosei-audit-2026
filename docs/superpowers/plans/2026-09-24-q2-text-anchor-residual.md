# Q2 Text-Anchor Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one `text_anchor_residual` Q2 fusion variant and evaluate it exactly once from A using Attachment 2 train/valid only, accepting only clean valid macro-F1 `>= 0.6212527658`.

**Architecture:** The new variant retains gated fusion unchanged, then runs the existing masked text projection through the same temporal encoder and pooling layer to make a text-only anchor. Its pooled state and `[text_coverage, 0, 0]` vector are added to the original fused representation. Audio and vision are never inputs to this anchor, and rows with no text receive an exact zero anchor.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers, pytest, local 7-Zip and local BERT assets.

---

### Task 1: Validate the New Fusion Variant

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing variant tests.**

Add `text_anchor_residual` to the supported-value parameterized config test. Add a model test constructing `MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0, fusion_variant="text_anchor_residual")` with the existing valid tensor fixture; require finite `logits[B,3]`, finite scores, and valid class argmaxes.

```python
model = MaskAwareTemporalFusion(
    hidden_size=16,
    heads=4,
    layers=1,
    dropout=0.0,
    fusion_variant="text_anchor_residual",
)
output = model(text=text, audio=audio, vision=vision, masks=masks)
assert output.logits.shape == (2, 3)
assert torch.isfinite(output.logits).all()
assert torch.isfinite(output.score).all()
predicted_classes = output.logits.argmax(dim=1)
assert torch.all((predicted_classes >= 0) & (predicted_classes <= 2))
```

- [ ] **Step 2: Verify RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_model.py -q
```

Expected: config validation rejects `text_anchor_residual` before model construction.

- [ ] **Step 3: Allow only the named variant.**

In `src/e_mosei_audit/q2/config.py`, extend the existing tuple exactly as follows:

```python
FUSION_VARIANTS = ("gated", "mag_lite", "mult_lite", "late_expert_shared", "text_anchor_residual")
```

Do not add a numerical text-anchor setting or change any default.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused command above. Commit only the config and focused config test update:

```bash
git add src/e_mosei_audit/q2/config.py tests/test_q2_config.py
git commit -m "feat: configure Q2 text anchor fusion"
```

### Task 2: Add the Mask-Safe Shared Text Anchor

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing mask and zero-anchor tests.**

Add a `text_anchor_residual` unavailable-value invariance test by replacing every unavailable raw text/audio/vision value with a large finite value and requiring unchanged logits, scores, gates, and temporal attention. Add two equal-weight, `dropout=0` comparisons against `gated`: with normal text availability, require at least one of logits or scores to differ, proving the anchor is nonzero; with all text masks false but each row temporally valid through audio/vision, require all public output tensors to be equal, proving the anchor is exactly zero when text is unavailable.

```python
anchor.load_state_dict(gated.state_dict(), strict=True)
text_missing_masks = TensorMasks(
    text=torch.zeros_like(masks.text),
    audio=masks.audio,
    vision=masks.vision,
    temporal=masks.temporal,
)
expected = gated(text=text, audio=audio, vision=vision, masks=text_missing_masks)
actual = anchor(text=text, audio=audio, vision=vision, masks=text_missing_masks)
assert torch.equal(actual.logits, expected.logits)
assert torch.equal(actual.score, expected.score)

with_text = anchor(text=text, audio=audio, vision=vision, masks=masks)
baseline = gated(text=text, audio=audio, vision=vision, masks=masks)
assert not (torch.equal(with_text.logits, baseline.logits) and torch.equal(with_text.score, baseline.score))
```

- [ ] **Step 2: Verify RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
```

Expected: the text-available comparison fails because `text_anchor_residual` still follows the identical gated path and therefore produces equal outputs.

- [ ] **Step 3: Implement the no-parameter anchor.**

After constructing the original `representation = torch.cat((pooled, availability_fraction), dim=1)`, add this exact branch before `_classify`:

```python
if self.fusion_variant == "text_anchor_residual":
    text_temporal = masks.temporal & availability[..., 0]
    pooled_text, _ = self._encode_late_expert(states[0], text_temporal)
    text_coverage = text_temporal.sum(dim=1, keepdim=True).to(text.dtype)
    text_coverage = text_coverage / masks.temporal.sum(dim=1, keepdim=True).to(text.dtype)
    text_anchor = torch.cat((pooled_text, text_coverage, text_coverage.new_zeros((text.shape[0], 2))), dim=1)
    representation = representation + text_anchor
```

`_encode_late_expert` already selects only rows that have at least one valid temporal slot and returns zero rows otherwise; reuse it rather than calling `temporal_encoder` with all-padding masks. Do not create new modules, parameters, loss terms, or a second BERT invocation.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused model command. Commit only the model and model tests:

```bash
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 text anchor residual fusion"
```

### Task 3: Prove Saved-Run and Train/Valid Boundaries

**Files:**
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write failing fake-archive tests.**

Using `TrainValidPayloadWithInaccessibleTest`, add a `run_q2` test with `fusion_variant="text_anchor_residual"`. Require a 30-row Attachment 3 prediction file, manifest `training.fusion_variant == "text_anchor_residual"`, and no test sentinel access. Add a saved checkpoint test constructed with the same fusion variant; monkeypatch `load_state_dict`, require `strict=True`, and use the same inaccessible test payload while calling `evaluate_saved_q2_valid`.

```python
assert manifest["training"]["fusion_variant"] == "text_anchor_residual"
assert len(predictions) == 30
assert observed_strict == [True]
```

- [ ] **Step 2: Verify the existing generic persistence path.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
```

Expected: PASS after Tasks 1 and 2, proving that the existing generic `fusion_variant` persistence/recovery boundary supports the newly validated variant without an extra runner code path.

- [ ] **Step 3: Reuse existing persistence paths without test loading.**

Do not modify `run_q2`, `evaluate_saved_q2_valid`, manifest writers, or data loaders: they already pass, persist, validate, and reconstruct `fusion_variant`, while historical manifests default to `gated`. Do not add `load_aligned_test`, test labels, test predictions, or test metrics.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused runner suite and full suite:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
```

Commit only the runner tests:

```bash
git add tests/test_q2_runner.py
git commit -m "test: cover Q2 text anchor valid-only runner"
```

### Task 4: Run and Audit Candidate J Once

**Files:**
- Create: `q2-text-anchor-residual.toml` (ignored local configuration)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-text-anchor-residual/` (ignored)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-text-anchor-residual.json` (ignored)
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Create the isolated config.**

Copy A's resolved fields exactly, set `output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-text-anchor-residual"` and `fusion_variant = "text_anchor_residual"`; retain `classification_variant="flat"`, `learning_rate=0.001`, `synthetic_missingness_enabled=false`, `text_adapter_variant="identity"`, and the fixed seed.

- [ ] **Step 2: Preflight and train exactly once.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-text-anchor-residual.toml --check
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-text-anchor-residual.toml
```

Require preflight `3395/728/30` and an absent output directory. Run the non-check command exactly once. Never access Attachment 2 test.

- [ ] **Step 3: Audit and decide.**

Require a nonempty model and manifest, 728 valid report support, 27 scenario rows, 30 Attachment 3 prediction rows, and a normalized A manifest difference of only `fusion_variant: gated -> text_anchor_residual`. Write the comparison JSON. Accept only macro-F1 `>=0.6212527658`; record an unassessed Test row, update the portfolio, run full pytest and `git diff --check`, then commit documentation.
