# Q2 Two-Point Valid Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit regression-loss coefficient and run the first fixed-seed valid-only candidate toward macro-F1 `>= 0.6212527658`.

**Architecture:** Keep the Q2 data/model interfaces unchanged. `Q2Config` owns a finite nonnegative `regression_loss_weight`; `runner` passes it to a pure joint-loss helper and records it in the run manifest. The candidate differs from v4 only by this coefficient and uses clean valid metrics only.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Transformers, pytest, local 7-Zip/BERT assets.

---

### Task 1: Make Joint-Loss Balance Explicit

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_runner.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing configuration and loss tests.**

Add `regression_loss_weight = 0.5` to valid TOML fixtures and direct `Q2Config(...)` helpers. Assert parsing exposes `0.5`, and a negative fixture raises `ValueError` containing `regression_loss_weight is outside its valid range`. Add a pure `_joint_loss` test that expects:

```python
output = Q2Output(
    logits=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
    score=torch.tensor([0.5, -0.25]),
    gates=torch.empty(0),
    temporal_attention=torch.empty(0),
)
labels = torch.tensor([0, 1])
scores = torch.tensor([0.0, -1.0])
class_weights = torch.tensor([1.0, 2.0, 1.0])
expected = nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
expected += 0.25 * nn.functional.smooth_l1_loss(output.score, scores)
assert torch.allclose(_joint_loss(output, labels, scores, class_weights, regression_loss_weight=0.25), expected)
```

Assert successful tiny-run manifests record `training.regression_loss_weight == 0.5`.

- [ ] **Step 2: Run focused tests to confirm RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py -q`.

Expected: failure because config/parser and `_joint_loss` lack the new field.

- [ ] **Step 3: Implement the minimal compatible change.**

Add the field to `Q2Config`, `_TRAINING_FIELDS`, parsing and validation. Implement:

```python
def _joint_loss(output, labels, scores, class_weights, *, regression_loss_weight):
    classification = nn.functional.cross_entropy(output.logits, labels, weight=class_weights)
    regression = nn.functional.smooth_l1_loss(output.score, scores)
    return classification + regression_loss_weight * regression
```

Pass the field through `run_q2` and `_train_epoch`; record it in the manifest. Add `regression_loss_weight = 0.5` to the example TOML. Do not change model/data/splits/class weights/missingness/test handling.

- [ ] **Step 4: Verify GREEN and commit.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py -q`, then run `git diff --check`.

Commit with `git commit -m "feat: configure Q2 regression loss weight"` after staging only the six files above.

### Task 2: Run and Audit the Classification-Priority Candidate

**Files:**
- Create (ignored local configuration): `q2.toml`
- Create (ignored artifact): `artifacts/q2-valid-regression-loss-025/`
- Create (ignored artifact): `artifacts/q2-valid-comparison-v4-regression-loss-025.json`
- Modify: `docs/superpowers/specs/2026-09-24-q2-two-point-valid-optimization-design.md`

- [ ] **Step 1: Write the candidate configuration.**

Use existing absolute archive, 7-Zip and BERT paths. Set output `/home/administrator/MyItem/E/artifacts/q2-valid-regression-loss-025`; preserve v4 values including `dropout=0.1`; set only `regression_loss_weight=0.25`.

- [ ] **Step 2: Preflight and train.**

Run `train-q2 --config q2.toml --check`, requiring counts `3395/728/30` and no output directory. Then run `train-q2 --config q2.toml` once.

- [ ] **Step 3: Audit and decide.**

Require identical archive/BERT/7-Zip/seed references to v4; the only training difference must be `regression_loss_weight`. Require a 728-sample report, 27 scenarios, 30 predictions, model and manifest. Write a valid-only comparison JSON. Stop if macro-F1 meets `0.6212527658`; otherwise update the design with the actual result before only the `hidden_size=256` candidate.

- [ ] **Step 4: Run full tests.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`. Expected: zero failures; only environment-gated real archive/Q1 tests may be skipped.
