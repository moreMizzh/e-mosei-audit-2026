# Q2 No-Train-Missingness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed-seed, valid-only Q2 candidate that disables only training-time synthetic contiguous modality loss while preserving v4 behavior as the explicit default.

**Architecture:** `Q2Config.synthetic_missingness_enabled` controls only the synthetic drop before each training batch. When false, `_train_epoch` forwards the observed batch masks in a `DroppedMasks(..., drops=())`; clean validation, the 27 validation missingness scenarios, and Attachment 3 inference retain their current paths. The run manifest records the switch inside the synthetic-missingness contract.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Transformers, pytest, local 7-Zip/BERT assets.

---

### Task 1: Make Training Missingness Explicit

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_q2_runner.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing configuration and train-path tests.**

Add `synthetic_missingness_enabled = true` to every valid TOML fixture and direct `Q2Config(...)` helper. Assert parsed configuration exposes `True`, and add a false TOML fixture that parses as `False`. Add a real tiny `run_q2` test with `synthetic_missingness_enabled=False` that monkeypatches `e_mosei_audit.q2.runner.apply_contiguous_drop` to raise `AssertionError("synthetic training drop should be disabled")`; the run must complete and its manifest must contain `manifest["training"]["synthetic_missingness"]["enabled"] is False`.

- [x] **Step 2: Run the focused suite and verify RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py -q`.

Expected: configuration parsing or direct construction fails because `synthetic_missingness_enabled` does not exist, and the disabled-path test cannot be satisfied.

- [x] **Step 3: Implement the smallest behavior change.**

Add `synthetic_missingness_enabled: bool` to `Q2Config` and `_TRAINING_FIELDS`; reject non-booleans in `_validate_training`. Pass it from `run_q2` into `_train_epoch` as a keyword-only boolean. In `_train_epoch`, retain the existing drop block when enabled and otherwise use the original observed batch masks without a synthetic drop:

```python
batch_masks = _slice_masks(masks, indexes)
if synthetic_missingness_enabled:
    count = int(rng.integers(1, 3))
    chosen = tuple(rng.choice(np.asarray(("text", "audio", "vision")), size=count, replace=False).tolist())
    dropped = apply_contiguous_drop(batch_masks, rng=rng, modalities=chosen)
else:
    dropped = DroppedMasks(masks=batch_masks, drops=())
```

Write the manifest as `"synthetic_missingness": {"enabled": config.synthetic_missingness_enabled, "modalities_per_sample": "1 or 2", "fraction_range": [0.1, 0.5]}`. Keep validation scenarios and Attachment 3 inference unchanged.

- [x] **Step 4: Verify GREEN and commit.**

Run the focused command, `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`, and `git diff --check`. Then stage only the six files above and commit `feat: configure Q2 training missingness`.

### Task 2: Run and Audit the Clean-Training Candidate

**Files:**
- Create (ignored local configuration): `q2.toml`
- Create (ignored artifact): `artifacts/q2-valid-no-train-missingness/`
- Create (ignored artifact): `artifacts/q2-valid-comparison-v4-no-train-missingness.json`
- Modify: `docs/superpowers/specs/2026-09-24-q2-two-point-valid-optimization-design.md`

- [x] **Step 1: Write the candidate configuration.**

Use the existing absolute archive, 7-Zip and BERT paths. Set `output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-no-train-missingness"`, `hidden_size = 128`, `dropout = 0.1`, `regression_loss_weight = 0.5`, `class_weight_exponent = 1.0`, and `synthetic_missingness_enabled = false`. Keep seed `20260924`, 30 epochs, batch size 64, learning rate 0.001, weight decay 0.0001, 4 heads, 2 layers and CUDA.

- [x] **Step 2: Preflight and train once.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml --check`, requiring `3395/728/30`; then run the same command without `--check` exactly once. Do not evaluate Attachment 2 test.

- [x] **Step 3: Audit and decide.**

Require same archive/BERT/7-Zip/seed/normalizer as v4; require only the training-missingness switch plus generated `best_epoch` to differ from v4. Require a 728-sample classification report, 27 validation-scenario rows, 30 Attachment 3 predictions, a nonempty model and run manifest. Write the comparison JSON. Stop if macro-F1 reaches `0.6212527658`; otherwise update the design with actual metrics before proposing a further candidate.

- [x] **Step 4: Run full tests.**

Result: clean macro-F1 `0.6165677806`, below the target but above v4 by `0.0153150148`; scenario mean/worst macro-F1 also improved. The next sequential candidate is the pre-registered polarity-intensity consistency loss in `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`.

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`; expect zero failures and only environment-gated real archive/Q1 skips.
