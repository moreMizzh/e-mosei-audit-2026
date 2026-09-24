# Q2 Availability-Conditioned Temporal Pooling Candidate Implementation Plan

> For agentic workers: execute this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and evaluate exactly one valid-only Q2 candidate that conditions existing temporal pool attention on local modality availability.

**Architecture:** Persist temporal_pooling_variant through fresh configs and manifests. The attention_availability model variant adds exactly three zero-initialized, bias-free availability-logit weights after the shared temporal encoder and before existing temporal masking/softmax. Historical manifests default missing field to attention.

**Tech Stack:** Python 3.12, PyTorch, pytest, TOML, existing e_mosei_audit.q2 runner.

---

### Task 1: Persist the temporal pooling variant

**Files:**
- Modify: src/e_mosei_audit/q2/config.py
- Modify: docs/q2-config.example.toml
- Modify: tests/test_q2_config.py
- Modify: tests/test_cli.py
- Modify: tests/test_q2_runner.py

- [x] **Step 1: Write failing configuration tests**

Add temporal_pooling_variant = "attention" to every valid inline Q2 TOML fixture. Add a parameterized parser test for exactly attention and attention_availability and an invalid value test asserting:

    temporal_pooling_variant must be one of: attention, attention_availability

Add attention to direct Q2Config constructors in shared CLI/runner fixtures. Add a missing-field test asserting:

    missing required training field: temporal_pooling_variant

- [x] **Step 2: Run configuration and fixture users to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py -q

Expected: red failures identify the missing strict field in parser/config constructors.

- [x] **Step 3: Implement exact config contract**

Add TEMPORAL_POOLING_VARIANTS = ("attention", "attention_availability"), temporal_pooling_variant to _TRAINING_FIELDS and frozen Q2Config, and:

    def validate_temporal_pooling_variant(value: object) -> str:
        if not isinstance(value, str) or value not in TEMPORAL_POOLING_VARIANTS:
            raise ValueError(
                "temporal_pooling_variant must be one of: attention, attention_availability"
            )
        return value

Validate it in _validate_training, parse/store it in load_q2_config, and document attention in the example TOML. Do not default a missing fresh configuration field.

- [x] **Step 4: Verify focused suites are green**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py -q

Expected: all focused configuration consumers pass.

- [x] **Step 5: Commit Task 1**

    git add src/e_mosei_audit/q2/config.py docs/q2-config.example.toml tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py
    git commit -m "feat: persist Q2 temporal pooling variant"

### Task 2: Implement zero-initialized availability-aware pooling

**Files:**
- Modify: src/e_mosei_audit/q2/model.py
- Modify: tests/test_q2_model.py

- [x] **Step 1: Write failing model tests**

Create matched attention and attention_availability models using identical seed. Assert the availability variant has exactly three additional parameters and only pool_availability_bias.weight as an additional state key; all shared parameter values and successor RNG values are identical. With zero bias, exact logits, score, gates, and temporal attention must equal A.

Use a recording pool bias to assert it receives only float availability shaped [B,T,3]. Add tests that all-invalid/padded slots retain zero attention, unavailable raw values remain inert, appended fully unavailable padding stays zero with existing narrow numerical tolerance only for cross-shape valid prefixes, and the three availability weights receive finite nonzero gradient in varied-availability batches.

- [x] **Step 2: Run model suite to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q

Expected: collection or assertions fail because the pooling variant and availability bias do not exist.

- [x] **Step 3: Implement the minimal pooling path**

Import validate_temporal_pooling_variant, accept temporal_pooling_variant: str = "attention" in the model constructor, and validate/store it. For attention_availability only, construct nn.Linear(3, 1, bias=False) after existing base modules inside a saved/restored torch RNG state, and set its weight to zero.

Replace only the attention-logit construction with:

    attention_logits = self.pool_attention(encoded).squeeze(-1)
    if self.temporal_pooling_variant == "attention_availability":
        attention_logits = attention_logits + self.pool_availability_bias(
            availability.to(dtype=encoded.dtype)
        ).squeeze(-1)
    attention_logits = attention_logits.masked_fill(~temporal, float("-inf"))

Do not change fusion, encoder, position feature, availability fraction, output heads, losses, optimizer, or late-expert path.

- [x] **Step 4: Verify model suite is green**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q

Expected: all model tests pass.

- [x] **Step 5: Commit Task 2**

    git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
    git commit -m "feat: add Q2 availability-aware temporal pooling"

### Task 3: Make run artifacts and saved-valid evaluation semantic

**Files:**
- Modify: src/e_mosei_audit/q2/runner.py
- Modify: tests/test_q2_runner.py

- [ ] **Step 1: Write failing runner tests**

Using the existing inaccessible-test payload, add an attention_availability config run test that asserts constructor propagation, a recorded training manifest field, 30 Attachment 3 rows, and zero forbidden test access. Add saved checkpoint tests showing strict reconstruction receives attention_availability when present and attention when absent from a historical manifest. Add an invalid manifest pooling variant rejection test.

- [ ] **Step 2: Run runner tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q

Expected: failures identify absent runner propagation or reconstruction semantics.

- [ ] **Step 3: Implement runner propagation and fallback**

Pass config.temporal_pooling_variant in normal model construction. Persist the field in run_manifest training. Add _manifest_temporal_pooling_variant(training) returning attention only when the field is absent and otherwise calling validate_temporal_pooling_variant. Pass its result into saved-valid model construction, preserving strict=True state loading.

- [ ] **Step 4: Verify runner suite is green**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py tests/test_cli.py -q

Expected: all runner/CLI tests pass and test-split sentinels remain unaccessed.

- [ ] **Step 5: Commit Task 3**

    git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
    git commit -m "feat: record Q2 temporal pooling artifacts"

### Task 4: Execute one N evaluation and record the decision

**Files:**
- Create: /home/administrator/MyItem/E/q2-attention-availability.toml (ignored local configuration)
- Create: artifacts/q2-valid-attention-availability/ (ignored runtime output)
- Modify: README.md
- Modify: docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md

- [ ] **Step 1: Run full regression verification**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
    git diff --check
    git status --short

Expected: all non-opt-in tests pass, no whitespace errors, and only planned files differ.

- [ ] **Step 2: Write the one-run configuration from A**

Create /home/administrator/MyItem/E/q2-attention-availability.toml with A's paths and all A training values, output_dir=/home/administrator/MyItem/E/artifacts/q2-valid-attention-availability, temporal_position_variant="none", and temporal_pooling_variant="attention_availability". Retain seed 20260924, fusion_variant="gated", and synthetic_missingness_enabled=false.

- [ ] **Step 3: Verify preflight and absent output**

Run:

    test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-attention-availability
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-attention-availability.toml --check

Expected: train_count=3395, valid_count=728, attachment3_count=30, with no output directory.

- [ ] **Step 4: Run exactly once and reconstruct saved valid**

Run exactly once:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-attention-availability.toml
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli evaluate-q2-valid --run-dir /home/administrator/MyItem/E/artifacts/q2-valid-attention-availability --output /home/administrator/MyItem/E/artifacts/q2-valid-attention-availability/recomputed_valid.json

Require exact saved/recomputed clean metrics, 728 valid rows, 27 scenario rows, 30 Attachment 3 predictions, manifest temporal_pooling_variant attention_availability, and no Attachment 2 test access.

- [ ] **Step 5: Record acceptance or retirement without test**

Write an A/N comparison JSON, update matching README Val and unassessed Test rows, and append the portfolio result. Accept only macro-F1 >= 0.6212527658. On failure, retire only the exact zero-initialized, three-weight attention_availability variant without tuning initialization, scale, bias, activation, inputs, loss, optimizer, checkpoint, or seed.

- [ ] **Step 6: Commit verified records**

    git add README.md docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
    git commit -m "docs: record Q2 availability pooling result"

## Plan Review

The four tasks cover configuration strictness, zero-initialized three-parameter pooling, mask/raw-value/padding/RNG invariants, valid-only runner persistence and legacy strict reconstruction, full test/preflight gates, exactly one N run, and valid-only reporting. The plan forbids test use, M modification, any sweep, and all tuning of the new pooling route.
