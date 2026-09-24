# Q2 Sinusoidal Temporal Position Candidate Implementation Plan

> For agentic workers: execute this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and evaluate one valid-only Q2 candidate whose only difference from control A is a fixed sinusoidal temporal position signal before the shared temporal encoder.

**Architecture:** Persist temporal_position_variant as a constrained Q2 training field. MaskAwareTemporalFusion will build parameter-free deterministic sinusoidal positions and add them only where its already-derived temporal mask is true, remasking padding before TransformerEncoder. The runner will pass and record the field and will reconstruct legacy manifests as none.

**Tech Stack:** Python 3.12, PyTorch, pytest, TOML, existing e_mosei_audit.q2 runner.

---

### Task 1: Persist the single candidate switch

**Files:**
- Modify: src/e_mosei_audit/q2/config.py:13-112,155-227
- Modify: docs/q2-config.example.toml:8-24
- Modify: tests/test_q2_config.py:10-114,288-333

- [x] **Step 1: Write failing config tests**

Add temporal_position_variant = "none" to every inline valid TOML fixture. Add a parameterized test requiring none and sinusoidal to load, and a test requiring unsupported to raise:

    with pytest.raises(
        ValueError,
        match=r"\Atemporal_position_variant must be one of: none, sinusoidal\Z",
    ):
        load_q2_config(config_path)

- [x] **Step 2: Run the new config tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q

Expected: failure identifies temporal_position_variant as absent from Q2Config, the exact TOML contract, or validation.

- [x] **Step 3: Implement the persisted field**

Add the field to _TRAINING_FIELDS and Q2Config; add TEMPORAL_POSITION_VARIANTS = ("none", "sinusoidal") and:

    def validate_temporal_position_variant(value: object) -> str:
        if not isinstance(value, str) or value not in TEMPORAL_POSITION_VARIANTS:
            raise ValueError("temporal_position_variant must be one of: none, sinusoidal")
        return value

Validate it from _validate_training, store it from load_q2_config, and document temporal_position_variant = "none" in the example TOML.

- [x] **Step 4: Verify config tests pass**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q

Expected: all config tests pass.

- [x] **Step 5: Commit the config contract**

    git add src/e_mosei_audit/q2/config.py docs/q2-config.example.toml tests/test_q2_config.py
    git commit -m "feat: persist Q2 temporal position variant"

### Task 2: Add the mask-safe, parameter-free position path

**Files:**
- Modify: src/e_mosei_audit/q2/model.py:12-16,86-105,220-231
- Modify: tests/test_q2_model.py:1-200,1280-1322

- [ ] **Step 1: Write failing model tests**

Import a desired _sinusoidal_position_encoding helper and assert its t=0 values are [0, 1, 0, 1, 0] for hidden width 5. Assert values at t=1 and t=2 follow the fixed 10000 base and the final odd dimension uses sine. Use an encoder recorder to assert none sends the pre-existing fused tensor, while sinusoidal differs only where temporal is true and padded slots remain zero.

Add model-pair tests that none and sinusoidal have exactly equal parameter counts and state-dict keys, constructing either leaves successor RNG values identical, unavailable raw values do not affect predictions, and appending fully masked padding does not affect predictions.

- [ ] **Step 2: Run the model tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q

Expected: import, constructor, or asserted encoder-input failure because the position feature does not exist.

- [ ] **Step 3: Implement only the fixed position path**

Import validate_temporal_position_variant, accept temporal_position_variant: str = "none" in MaskAwareTemporalFusion.__init__, and store its validated value. Add a helper with torch.arange, even-index sine, odd-index cosine, no nn.Parameter, no buffer, and no random draw.

Immediately before the existing shared encoder call, apply:

    if self.temporal_position_variant == "sinusoidal":
        positions = _sinusoidal_position_encoding(
            fused.shape[1], fused.shape[2], device=fused.device, dtype=fused.dtype
        )
        fused = fused + positions.unsqueeze(0) * temporal.unsqueeze(-1).to(dtype=fused.dtype)
        fused = fused.masked_fill(~temporal.unsqueeze(-1), 0.0)

Do not alter late-expert behavior, fusion variants, optimizer, loss, checkpoint selection, or masks.

- [ ] **Step 4: Verify model tests pass**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q

Expected: all model tests pass, including position, RNG, unavailable-value, and padding invariants.

- [ ] **Step 5: Commit the model behavior**

    git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
    git commit -m "feat: add Q2 sinusoidal temporal positions"

### Task 3: Propagate the variant through valid-only runner artifacts

**Files:**
- Modify: src/e_mosei_audit/q2/runner.py:191-199,310-331,744-775
- Modify: tests/test_q2_runner.py:360-520,1270-1381,1509-1561
- Modify: tests/test_cli.py:184-240

- [ ] **Step 1: Write failing runner and CLI tests**

Add a fake train/valid payload whose test lookup raises. In a temporal_position_variant="sinusoidal" run, assert run_manifest.json records the field and the model constructor receives it; assert archive verification is one and forbidden test lookup was never attempted.

Create a saved sinusoidal checkpoint and manifest. Monkeypatch the constructor/load to assert temporal_position_variant == "sinusoidal" and strict is True. Create a historical manifest omitting the field and assert strict reconstruction receives none. Update each Q2 config test fixture and CLI fixture to include none.

- [ ] **Step 2: Run runner tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py tests/test_cli.py -q

Expected: failure because run construction and saved-valid reconstruction lack the temporal-position field.

- [ ] **Step 3: Implement runner propagation and legacy fallback**

Pass config.temporal_position_variant from run_q2 into MaskAwareTemporalFusion and store it in manifest["training"]. In saved-valid reconstruction, call _manifest_temporal_position_variant(training), returning none if absent and otherwise invoking validate_temporal_position_variant; pass the result into the reconstructed model.

- [ ] **Step 4: Verify runner/CLI tests pass**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py tests/test_cli.py -q

Expected: all runner/CLI tests pass while test-split sentinels remain unaccessed.

- [ ] **Step 5: Commit valid-only artifact behavior**

    git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py tests/test_cli.py
    git commit -m "feat: record Q2 temporal position artifacts"

### Task 4: Gate and execute exactly one M evaluation

**Files:**
- Create: artifacts/q2-valid-sinusoidal-temporal-position/ (ignored runtime output only)
- Modify: README.md:3-54 after the actual result exists
- Modify: docs/portfolio/q2-valid-experiment-records.md after the actual result exists

- [ ] **Step 1: Run complete regression verification and review the exact diff**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
    git diff --check
    git status --short

Expected: all non-opt-in tests pass, no whitespace errors, and only planned files differ.

- [ ] **Step 2: Create M config from A with exactly one training difference**

Copy the exact control-A local config. Set only:

    output_dir = "artifacts/q2-valid-sinusoidal-temporal-position"
    temporal_position_variant = "sinusoidal"

Retain seed=20260924, fusion_variant="gated", and synthetic_missingness_enabled=false. Do not create a test target.

- [ ] **Step 3: Verify preflight and output absence**

Run:

    test ! -e artifacts/q2-valid-sinusoidal-temporal-position
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-sinusoidal-temporal-position.toml --check

Expected: train_count=3395, valid_count=728, attachment3_count=30, without output creation.

- [ ] **Step 4: Execute exactly one M run and reconstruct saved valid predictions**

Run exactly once:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-sinusoidal-temporal-position.toml
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli evaluate-q2-valid --run-dir artifacts/q2-valid-sinusoidal-temporal-position --output artifacts/q2-valid-sinusoidal-temporal-position/recomputed_valid.json

Compare saved and recomputed clean metrics. Require 728 valid predictions, 27 scenarios, 30 Attachment 3 rows, temporal_position_variant="sinusoidal" in the manifest, and no test member in the run path.

- [ ] **Step 5: Record the actual decision without test evaluation**

Write the A/M comparison JSON, update only the README Val row and Test 未评估 row, and append the portfolio record. Accept only if clean valid macro-F1 is at least 0.6212527658; otherwise record actual metrics and permanently retire this exact fixed variant without tuning it.

- [ ] **Step 6: Commit documentation only after result verification**

    git add README.md docs/portfolio/q2-valid-experiment-records.md
    git commit -m "docs: record Q2 sinusoidal position result"

## Plan Review

The four tasks cover the approved specification: exact enum/config contract, deterministic odd-width sinusoid, zero parameter/RNG invariants, temporal-only mask application, normal training propagation, historical strict reconstruction, test-split sentinel protection, full-suite verification, one real valid-only run, artifact reconstruction, and result reporting. No task introduces a learnable position table, a hyperparameter sweep, a second seed, a synthetic-missingness change, or test access.
