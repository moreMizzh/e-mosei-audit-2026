# Q2 CORN Ordinal Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one `corn` classification variant to the Q2 gated model and evaluate it once from A using Attachment 2 train/valid only, accepting only clean valid macro-F1 `>= 0.6212527658`.

**Architecture:** `flat` remains the default three-logit classifier. `corn` replaces only its classification parameterization with two conditional ordinal logits; their chain-rule probabilities are written as three log-probabilities into the existing `Q2Output.logits` interface. Training selects CE for `flat` and conditional BCE for `corn`; the regression head, masks, fusion, optimizer and validation paths stay unchanged.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers, pytest, local 7-Zip and local BERT assets.

---

### Task 1: Persist and Validate the Classification Variant

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write failing config tests.**

Add `classification_variant = "flat"` to every existing valid TOML fixture and direct `Q2Config` test helper. Add a parametrized TOML parse test for `flat` and `corn`; add an unsupported-value test requiring exactly `classification_variant must be one of: flat, corn`. Make the fake runner configuration explicitly use `classification_variant="flat"`.

- [ ] **Step 2: Verify RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py -q
```

Expected: valid fixtures reject the unexpected field and the new variant parser test fails because the field and validator do not exist.

- [ ] **Step 3: Add only the persisted enum contract.**

In `config.py`, add the field to every persistence boundary:

```python
CLASSIFICATION_VARIANTS = ("flat", "corn")

# Add "classification_variant" after "text_adapter_variant" in _TRAINING_FIELDS.
# Add classification_variant: str to Q2Config and pass
# validate_classification_variant(values["classification_variant"]) from load_q2_config.

def validate_classification_variant(value: object) -> str:
    if not isinstance(value, str) or value not in CLASSIFICATION_VARIANTS:
        raise ValueError("classification_variant must be one of: flat, corn")
    return value
```

Call the validator from `_validate_training`; add `classification_variant = "flat"` to the Q2 example TOML. Do not add a numerical CORN hyperparameter or a default implicit in a new training TOML.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused suite above. Commit only the configuration, fixture and template changes as `feat: configure Q2 ordinal classification variant`.

### Task 2: Implement the Rank-Consistent Output and Loss

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `tests/test_q2_model.py`
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write failing model and loss tests.**

Create `MaskAwareTemporalFusion(..., classification_variant="corn")` using the existing valid input fixture. Require finite `output.logits[B,3]`, finite `output.ordinal_logits[B,2]`, `torch.exp(output.logits).sum(dim=1)==1`, class indices in `[0,2]`, bounded scores, and invariant predictions after filling unavailable raw modality values with a large number.

Add a loss test with `ordinal_logits=[[a0,a1],[b0,b1],[c0,c1]]`, labels `[0,1,2]`, nonuniform three-class weights, and zero regression/consistency weights. Compute each conditional BCE with `reduction="none"`, multiply by `class_weights[labels]`, divide by that participating subset's weight sum, then average the two task losses; assert `_joint_loss` equals it. Add an all-negative batch case proving the weighted first conditional BCE is finite when the second conditional task has no samples.

- [ ] **Step 2: Verify RED.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py tests/test_q2_runner.py -q
```

Expected: construction rejects `classification_variant`, and no ordinal output/loss is present.

- [ ] **Step 3: Implement the minimal CORN path.**

Import `validate_classification_variant`. Preserve `expert_weights` as the fifth positional `Q2Output` field and append `ordinal_logits` after it:

```python
@dataclass(frozen=True)
class Q2Output:
    logits: torch.Tensor
    score: torch.Tensor
    gates: torch.Tensor
    temporal_attention: torch.Tensor
    expert_weights: torch.Tensor | None = None
    ordinal_logits: torch.Tensor | None = None
```

Accept and validate `classification_variant` in the model constructor. Use `nn.Linear(hidden_size + 3, 3)` for `flat` and `nn.Linear(hidden_size + 3, 2)` for `corn`. Reject the unsupported combination `classification_variant="corn"` with `fusion_variant="late_expert_shared"` before constructing heads, since expert-logit interpolation is only defined for the flat three-logit head.

For gated/MAG/MulT representations, implement a private classification helper. For CORN it must use:

```python
ordinal_logits = self.classifier(representation)
conditional = torch.sigmoid(ordinal_logits)
probabilities = torch.stack(
    (1.0 - conditional[:, 0], conditional[:, 0] * (1.0 - conditional[:, 1]), conditional[:, 0] * conditional[:, 1]),
    dim=1,
)
logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
```

Return both `logits` and `ordinal_logits`; flat returns the direct three logits and `None`. Keep score, gates, temporal attention, masking and all flat state-dict keys unchanged.

In `runner.py`, add a private classification-loss helper called by `_joint_loss`:

```python
if output.ordinal_logits is None:
    classification = nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
else:
    sample_weights = class_weights[labels]
    first_terms = nn.functional.binary_cross_entropy_with_logits(
        output.ordinal_logits[:, 0], (labels > 0).to(output.ordinal_logits.dtype), reduction="none"
    )
    first = (first_terms * sample_weights).sum() / sample_weights.sum()
    active = labels > 0
    terms = [first]
    if bool(active.any()):
        second_weights = sample_weights[active]
        second_terms = nn.functional.binary_cross_entropy_with_logits(
            output.ordinal_logits[active, 1], (labels[active] > 1).to(output.ordinal_logits.dtype), reduction="none"
        )
        terms.append((second_terms * second_weights).sum() / second_weights.sum())
    classification = torch.stack(terms).mean()
```

Keep the existing regression and optional polarity-consistency terms exactly after `classification`.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused model/runner suites. Commit the model, runner and tests as `feat: add Q2 CORN ordinal classification`.

### Task 3: Preserve Saved-Run and Train/Valid Boundaries

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write failing fake-archive tests.**

Add one tiny `run_q2` test with `classification_variant="corn"` and the existing pickle round-trip `test` sentinel. Require output `728`-analogue fake support, 30 fake attachment-3 predictions, manifest field `classification_variant="corn"`, and no sentinel access. Add a saved CORN checkpoint test through `evaluate_saved_q2_valid`; require the manifest variant rebuilds the two-logit head and calls `load_state_dict(..., strict=True)`. Add an unsupported saved-manifest classification variant test.

- [ ] **Step 2: Verify RED.**

Run the new focused node ids. Expected: runner omits the new constructor and manifest field, while saved reconstruction cannot select `corn`.

- [ ] **Step 3: Wire persistence without touching test.**

Pass `classification_variant=config.classification_variant` when constructing training models. Save it in the manifest's `training` mapping. Add `_manifest_classification_variant(training)` that returns `flat` when historical manifests omit the key, otherwise calls `validate_classification_variant`; pass it during saved valid reconstruction. Continue to use `load_aligned_train_valid` only; do not introduce a test split loader.

- [ ] **Step 4: Verify GREEN and commit.**

Run `tests/test_q2_runner.py -q`, then the full suite. Commit runner persistence and tests as `test: cover Q2 CORN valid-only runner`.

### Task 4: Run and Audit Candidate I Once

**Files:**
- Create: `q2-corn.toml` (ignored local configuration)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-corn/` (ignored)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-corn.json` (ignored)
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [x] **Step 1: Create the isolated config.**

Copy A's resolved fields exactly, set `output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-corn"` and `classification_variant = "corn"`; keep `learning_rate=0.001`, `synthetic_missingness_enabled=false`, `fusion_variant="gated"`, `text_adapter_variant="identity"`, all other fields and fixed seed unchanged.

- [x] **Step 2: Preflight and train exactly once.**

Run `train-q2 --config q2-corn.toml --check`, requiring `3395/728/30` and no output directory. Then run the same command once without `--check`. Never access or report Attachment 2 test.

- [x] **Step 3: Audit and decide.**

Require nonempty model/manifest, four clean metrics, 728 valid report support, 27 scenario rows and 30 attachment-3 rows. Normalize A's missing historical fields plus `classification_variant="flat"`; require every field and normalizer equality except `classification_variant: flat -> corn`. Write the comparison JSON and accept only macro-F1 `>=0.6212527658`. Add Val/Test ledger rows with Test unassessed, update the portfolio, run full pytest and `git diff --check`, then commit documentation.
