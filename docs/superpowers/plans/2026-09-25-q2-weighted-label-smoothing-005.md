# Q2 Weighted Label-Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one persisted fixed weighted-label-smoothing loss and valid-only evaluate it exactly once as an A-derived Q2 candidate.

**Architecture:** A new config semantic selects hard CE or PyTorch's fixed `label_smoothing=0.05`; all model, mask, feature, optimizer, checkpoint, and inference code remains unchanged. The runner writes and validates the semantic in saved manifests so strict valid reconstruction cannot accept an unsupported candidate.

**Tech Stack:** Python 3.12, PyTorch, pytest, TOML, current `e_mosei_audit.q2` config/runner, offline BERT, local 7-Zip.

---

## File Map

- `src/e_mosei_audit/q2/config.py`: persisted enum, new-config validation, incompatible-combination guard, `Q2Config` field.
- `src/e_mosei_audit/q2/runner.py`: loss dispatch, train wiring, manifest persistence and strict replay validation.
- `tests/test_q2_config.py`: exact parsing and invalid-combination contracts.
- `tests/test_q2_runner.py`: loss equation, runner test-isolation, persistence and strict replay.
- `README.md` and `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`: real result only after the one run.

### Task 1: Persist And Validate The Loss Semantic

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py:18-80,100-128,196-220,246-280`
- Modify: `tests/test_q2_config.py:20-70,360-530,860-950`

- [ ] **Step 1: Write failing config tests**

Add `classification_loss_variant = "hard_ce"` to every complete TOML fixture; exact-field validation must remain strict. Add this parameterization:

```python
@pytest.mark.parametrize("variant", ["hard_ce", "weighted_label_smoothing_005"])
def test_load_q2_config_parses_supported_classification_loss_variant(
    tmp_path: Path, variant: str
) -> None:
    config_path = write_complete_q2_toml(tmp_path, classification_loss_variant=variant)
    assert load_q2_config(config_path).classification_loss_variant == variant
```

Add an unsupported-value test requiring `classification_loss_variant must be one of: hard_ce, weighted_label_smoothing_005`. Add complete fixtures for `classification_variant="corn"` plus smoothing and for `dropout_consistency_variant="rdrop_alpha_1"` plus smoothing, requiring respectively:

```python
"weighted_label_smoothing_005 requires classification_variant=flat"
"weighted_label_smoothing_005 cannot be combined with rdrop_alpha_1"
```

- [ ] **Step 2: Prove the new tests are red**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q
```

Expected: new accepted-field tests fail because `classification_loss_variant` is not recognized; failures must not be TOML syntax errors.

- [ ] **Step 3: Implement the smallest config contract**

Add this enum, required field, load result, and validator:

```python
CLASSIFICATION_LOSS_VARIANTS = ("hard_ce", "weighted_label_smoothing_005")

def validate_classification_loss_variant(value: object) -> str:
    if not isinstance(value, str) or value not in CLASSIFICATION_LOSS_VARIANTS:
        raise ValueError(
            "classification_loss_variant must be one of: hard_ce, weighted_label_smoothing_005"
        )
    return value
```

Insert `classification_loss_variant` immediately after `classification_variant` in `_TRAINING_FIELDS` and `Q2Config`. Add:

```python
def validate_classification_loss_training(
    value: object, *, classification_variant: str, dropout_consistency_variant: str
) -> str:
    variant = validate_classification_loss_variant(value)
    if variant != "weighted_label_smoothing_005":
        return variant
    if classification_variant != "flat":
        raise ValueError("weighted_label_smoothing_005 requires classification_variant=flat")
    if dropout_consistency_variant != "none":
        raise ValueError("weighted_label_smoothing_005 cannot be combined with rdrop_alpha_1")
    return variant
```

In `_validate_training`, reuse the validated classification and dropout-consistency values, retain the R-Drop/dropout checks, then call `validate_classification_loss_training`. Do not alter class weights, model fields, data, or defaults.

- [ ] **Step 4: Prove config GREEN and commit**

Run the Step 2 command; it must pass. Then:

```bash
git add src/e_mosei_audit/q2/config.py tests/test_q2_config.py
git commit -m "feat: configure Q2 weighted label smoothing"
```

### Task 2: Add The Exact Loss Dispatch

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py:620-755,840-975`
- Modify: `tests/test_q2_runner.py:45-200,560-680,900-940`

- [ ] **Step 1: Write failing loss and fake-run tests**

Add a direct loss test using nonuniform weights and flat logits:

```python
def test_classification_loss_uses_fixed_weighted_label_smoothing() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.2, 0.4, 1.5]], requires_grad=True)
    output = Q2Output(logits, torch.zeros(2), torch.empty(0), torch.empty(0), None)
    labels = torch.tensor([0, 2])
    weights = torch.tensor([1.0, 2.0, 3.0])
    actual = _classification_loss(
        output, labels, weights, classification_loss_variant="weighted_label_smoothing_005"
    )
    expected = torch.nn.functional.cross_entropy(
        logits, labels, weight=weights, label_smoothing=0.05
    )
    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
```

Change existing hard-CE direct calls to explicit `classification_loss_variant="hard_ce"` and preserve their exact prior equality. Add a fake-archive run using only this `replace` difference:

```python
config = replace(
    runner_config(tmp_path),
    output_dir=tmp_path / "q2-label-smoothing-output",
    classification_loss_variant="weighted_label_smoothing_005",
)
```

Its Attachment 2 test accessor remains inaccessible; require 30 Attachment 3 predictions and persisted `training["classification_loss_variant"]`.

- [ ] **Step 2: Prove runner RED**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
```

Expected: `_classification_loss` rejects the new keyword before implementation and the new config field is unavailable until Task 1 lands.

- [ ] **Step 3: Implement only the loss call chain**

Add `classification_loss_variant` to `_train_epoch`, `_joint_loss`, `_rdrop_joint_loss`, and `_classification_loss`; pass `config.classification_loss_variant` from `run_q2` to every train epoch. Implement the flat branch exactly:

```python
variant = validate_classification_loss_variant(classification_loss_variant)
if output.ordinal_logits is None:
    if variant == "hard_ce":
        return nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
    if variant == "weighted_label_smoothing_005":
        return nn.functional.cross_entropy(
            output.logits, labels, weight=class_weights, label_smoothing=0.05
        )
    raise AssertionError(f"unreachable classification loss variant: {variant}")
if variant != "hard_ce":
    raise ValueError("non-flat classification requires classification_loss_variant=hard_ce")
```

Preserve the existing CORN conditional-BCE branch verbatim. Persist `classification_loss_variant` next to `classification_variant` in the run manifest. Update factories and direct test calls with `hard_ce`. Do not change `_evaluate`, prediction argmax, model, masks, optimizer, or checkpoint selection.

- [ ] **Step 4: Prove runner GREEN and commit**

Run the Step 2 command; it must pass. Then:

```bash
git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
git commit -m "feat: add Q2 weighted label smoothing loss"
```

### Task 3: Validate Saved Manifests And Strict Replay

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py:350-525`
- Modify: `tests/test_q2_runner.py:1450-2460`

- [ ] **Step 1: Write failing replay tests**

Construct a strict saved-valid fake run with flat state dict and these manifest values:

```python
"classification_variant": "flat",
"classification_loss_variant": "weighted_label_smoothing_005",
"dropout_consistency_variant": "none",
```

Its fake Attachment 2 test accessor raises; require `evaluate_saved_q2_valid` to use only valid. Duplicate with `classification_loss_variant="unsupported"` and require the exact enum `ValueError`. Preserve a field-absent historical manifest test and require its hard-CE fallback.

- [ ] **Step 2: Prove strict replay RED**

Run only the new test node IDs. Expected: the unsupported manifest is accepted before saved-manifest validation exists, proving the test reaches the intended boundary.

- [ ] **Step 3: Implement fallback and early validation**

Add:

```python
def _manifest_classification_loss_variant(training: Mapping[str, object]) -> str:
    if "classification_loss_variant" not in training:
        return "hard_ce"
    return validate_classification_loss_variant(training["classification_loss_variant"])
```

Before archive access in `evaluate_saved_q2_valid`, obtain the classification, dropout-consistency, and loss semantics; call `validate_classification_loss_training` with them. Keep this value out of inference math. Pass the already-validated classification variant into model construction.

- [ ] **Step 4: Prove full GREEN and commit**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
```

All commands must pass before:

```bash
git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
git commit -m "test: validate Q2 label smoothing saved runs"
```

### Task 4: One Valid-Only Run And Frozen Record

**Files:**
- Create ignored: `/home/administrator/MyItem/E/q2-label-smoothing-005.toml`
- Create ignored: `/home/administrator/MyItem/E/artifacts/q2-valid-label-smoothing-005/`
- Create ignored: `/home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-label-smoothing-005.json`
- Modify: `README.md:7-65`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`
- Modify: this plan

- [ ] **Step 1: Preflight an A-normalized ignored TOML**

Copy every A path/training value exactly, except use:

```toml
output_dir = "artifacts/q2-valid-label-smoothing-005"
classification_loss_variant = "weighted_label_smoothing_005"
```

It must retain seed `20260924`, flat/gated/identity/last-hidden-state, no synthetic missingness, `rdrop_alpha_1` disabled, position `none`, and temporal attention. Require it and its output to be ignored, then run only:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2-label-smoothing-005.toml --check
```

Require `train=3395`, `valid=728`, `attachment3=30`, and no created output. Do not invoke any Test command.

- [ ] **Step 2: Run training exactly once**

After all Task 3 gates pass, run once with the same command without `--check`. Do not edit/rerun config or invoke Attachment 2 Test.

- [ ] **Step 3: Rebuild valid exactly once and compare**

Run once with a fresh report path:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli evaluate-q2-valid --run-dir artifacts/q2-valid-label-smoothing-005 --output artifacts/q2-valid-label-smoothing-005/strict-valid-report.json
```

Require saved/rebuilt clean Accuracy, macro-F1, MAE, and Pearson to match; valid support 728; 27 scenario rows; 30 Attachment 3 predictions; nonempty model; and a normalized A/Q manifest difference containing only `classification_loss_variant`. Write the ignored comparison JSON with this evidence and an explicit train/valid-only scope.

- [ ] **Step 4: Record and freeze**

Add exactly one paired README Val/Test row: Val uses real metrics; Test is all `-` and `未评估`. Record metrics, class F1, scenario mean/worst, strict replay, and the sole manifest difference in the exploration portfolio. Accept only macro-F1 `>= 0.6212527658`; otherwise retire exactly epsilon `0.05`, with no epsilon/class-weight/loss-weight/checkpoint/seed/test/combination tuning. Run full tests and `git diff --check`, then commit the three documentation files with `docs: record Q2 label smoothing result`.
