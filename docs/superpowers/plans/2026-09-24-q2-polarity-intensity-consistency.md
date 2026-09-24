# Q2 Polarity-Intensity Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit, fixed-weight consistency loss between the existing three-class polarity head and regression intensity head, then run it once on the promoted no-training-missingness control.

**Architecture:** The current regression target is in `[-3, 3]`, while the expected class polarity of `softmax(logits)` over `[-1, 0, 1]` lies in `[-1, 1]`. The new loss is `SmoothL1(score / 3, expected_polarity)` and enters the existing joint loss with explicit nonnegative `polarity_consistency_loss_weight`. Default `0.0` exactly preserves current behavior; the one candidate uses fixed `0.10`.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Transformers, pytest, local 7-Zip/BERT assets.

---

### Task 1: Make Head Consistency Explicit

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_runner.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing configuration, loss, and train-wiring tests.**

Add required `polarity_consistency_loss_weight = 0.0` to every valid TOML fixture and direct `Q2Config(...)` helper. Assert it parses; assert negative and NaN values raise `ValueError` containing `polarity_consistency_loss_weight is outside its valid range`.

Extend the pure joint-loss test with:

```python
probabilities = torch.softmax(output.logits, dim=1)
expected_polarity = probabilities @ torch.tensor([-1.0, 0.0, 1.0])
expected += 0.10 * torch.nn.functional.smooth_l1_loss(output.score / 3.0, expected_polarity)
```

Then assert `_joint_loss(..., regression_loss_weight=0.25, polarity_consistency_loss_weight=0.10)` equals `expected`. Add a real tiny `run_q2` wrapper around `_joint_loss` using config `polarity_consistency_loss_weight=0.10`, delegate to the original helper, record its keyword argument and assert every train call receives `0.10`; verify the manifest records `0.10`.

- [ ] **Step 2: Run focused tests and observe RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py -q`.

Expected: absent Q2Config argument/config field and absent joint-loss keyword cause failure.

- [ ] **Step 3: Implement the minimal compatible change.**

Add a finite nonnegative float `polarity_consistency_loss_weight` to Q2 config validation and parsing. In `runner.py`, define class polarities on the logits device/dtype and add this exact term:

```python
probabilities = torch.softmax(output.logits, dim=1)
expected_polarity = probabilities @ output.logits.new_tensor([-1.0, 0.0, 1.0])
consistency = nn.functional.smooth_l1_loss(output.score / 3.0, expected_polarity)
return classification + regression_loss_weight * regression + polarity_consistency_loss_weight * consistency
```

Pass the config field from `run_q2` through `_train_epoch` into `_joint_loss`; write it under `training.polarity_consistency_loss_weight` in the manifest. Do not change model architecture, labels, masks, epoch selection, validation scenarios or Attachment 3 inference.

- [ ] **Step 4: Verify GREEN and commit.**

Run the focused command, full `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`, and `git diff --check`. Commit only the six files with `git commit -m "feat: configure Q2 polarity consistency loss"`.

### Task 2: Run and Audit the Consistency Candidate

**Files:**
- Create (ignored local configuration): `q2.toml`
- Create (ignored artifact): `artifacts/q2-valid-polarity-consistency-010/`
- Create (ignored artifact): `artifacts/q2-valid-comparison-no-train-missingness-consistency-010.json`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Write the one-variable configuration.**

Use A's values: `hidden_size=128`, `dropout=0.1`, `regression_loss_weight=0.5`, `class_weight_exponent=1.0`, and `synthetic_missingness_enabled=false`. Set only `output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-polarity-consistency-010"` and `polarity_consistency_loss_weight = 0.10`.

- [ ] **Step 2: Preflight and train once.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml --check`, requiring `3395/728/30`, then run the same command without `--check` exactly once. Do not evaluate Attachment 2 test.

- [ ] **Step 3: Audit and decide.**

Require A-equivalent archive/BERT/7-Zip/seed/normalizer and all training values except the new fixed consistency weight; require 728 valid report entries, 27 scenario rows, 30 Attachment 3 predictions, model and manifest. Accept only if macro-F1 reaches `0.6212527658`, Neutral F1 is at least A's `0.4890109890`, and MAE is no more than v4 MAE `+0.01`. Otherwise record the result and move to MAG-lite, not loss-weight tuning.

- [ ] **Step 4: Run full tests.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`; expect zero failures and only environment-gated real archive/Q1 skips.
