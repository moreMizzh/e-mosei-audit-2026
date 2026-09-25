# Q2 Availability-Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one persisted zero-initialized availability embedding and evaluate
exactly one valid-only A-derived Candidate T run.

**Architecture:** `temporal_context_variant="availability_embedding"` adds a
3-to-hidden, bias-free zero-initialized linear map of existing modality
availability to the gated fused state before the temporal Transformer. The
default `none` path remains identical; runner validation, manifests, and strict
replay persist the semantic.

**Tech Stack:** Python 3.12, PyTorch, pytest, TOML, current
`e_mosei_audit.q2` config/model/runner, local offline BERT and 7-Zip.

---

## File Map

- `src/e_mosei_audit/q2/config.py`: required context enum, Q2Config field, and
  early late-expert compatibility validation.
- `src/e_mosei_audit/q2/model.py`: zero-init availability embedding and the
  only new fused-state addition.
- `src/e_mosei_audit/q2/runner.py`: no-I/O validation, model construction,
  manifest persistence, strict-replay fallback and validation.
- `docs/q2-config.example.toml`: explicit default `temporal_context_variant`.
- `tests/test_q2_config.py`, `tests/test_q2_model.py`,
  `tests/test_q2_runner.py`, `tests/test_cli.py`: complete persisted contract,
  model safety, fake-run Test isolation, and replay coverage.
- `README.md` and `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`:
  actual results only after the single run.

## Task 1: Persist The Context Contract

- [ ] **Step 1: Write failing config/model tests.**

Add `temporal_context_variant = "none"` to all complete inline TOML fixtures
and direct `Q2Config` constructors. Parameterize acceptance for `none` and
`availability_embedding`; require unknown input to fail with:

```python
"temporal_context_variant must be one of: none, availability_embedding"
```

Add an explicit `fusion_variant="late_expert_shared"` plus
`temporal_context_variant="availability_embedding"` fixture requiring:

```python
"availability_embedding temporal context is unsupported with late_expert_shared fusion"
```

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python \
  -m pytest tests/test_q2_config.py tests/test_q2_model.py -q
```

Expected RED: missing `Q2Config` field, unexpected TOML field, or unsupported
variant errors caused by the absent context implementation.

- [ ] **Step 2: Implement the minimal contract.**

In `config.py`, add:

```python
TEMPORAL_CONTEXT_VARIANTS = ("none", "availability_embedding")

def validate_temporal_context_variant(value: object) -> str:
    if not isinstance(value, str) or value not in TEMPORAL_CONTEXT_VARIANTS:
        raise ValueError("temporal_context_variant must be one of: none, availability_embedding")
    return value

def validate_temporal_context_training(value: object, *, fusion_variant: str) -> str:
    variant = validate_temporal_context_variant(value)
    if variant == "availability_embedding" and fusion_variant == "late_expert_shared":
        raise ValueError("availability_embedding temporal context is unsupported with late_expert_shared fusion")
    return variant
```

Insert the required field after `temporal_position_variant` in `_TRAINING_FIELDS`
and `Q2Config`; use the training validator from `_validate_training` after the
validated fusion variant is known; parse it into `Q2Config`. Add the default to
the example TOML. Do not add optional defaults to a fresh configuration.

- [ ] **Step 3: Verify green and commit.**

Rerun the Step 1 command, require it to pass, then commit only config/template
and test changes:

```bash
git add src/e_mosei_audit/q2/config.py docs/q2-config.example.toml tests/test_q2_config.py tests/test_q2_model.py
git commit -m "feat: configure Q2 availability context"
```

## Task 2: Add The Zero-Initialized Fused Context

- [ ] **Step 1: Write RED model tests.**

For matched `none` and `availability_embedding` models initialized from the
same seed, require all pre-existing state entries and the successor
`torch.rand` values to be exact matches; require the sole new entry
`availability_embedding.weight` to have shape `[hidden_size, 3]` and be all
zeros. Feed identical mixed availability inputs and assert equality for logits,
score, gates, and temporal attention before optimization.

Wrap unavailable raw tensor entries and appended masked padding in
`1_000_000.0` and require all public outputs to remain equal. Backpropagate
`output.logits.square().sum() + output.score.square().sum()` on an available
batch and require finite nonzero `availability_embedding.weight.grad`.

Run focused new model node IDs. Expected RED: the constructor rejects the
variant or the zero-init equality/gradient checks fail because no embedding
exists.

- [ ] **Step 2: Implement only the new context branch.**

Import the context validator. In `MaskAwareTemporalFusion.__init__`, store the
validated field and reject late expert via the same exact error. For the new
variant create the sole new module while restoring RNG state:

```python
if self.temporal_context_variant == "availability_embedding":
    rng_state = torch.get_rng_state()
    try:
        self.availability_embedding = nn.Linear(3, hidden_size, bias=False)
        nn.init.zeros_(self.availability_embedding.weight)
    finally:
        torch.set_rng_state(rng_state)
```

Immediately after the existing gated weighted sum and before its existing
`masked_fill`, add:

```python
if self.temporal_context_variant == "availability_embedding":
    fused = fused + self.availability_embedding(availability.to(dtype=fused.dtype))
```

Do not add it to the late-expert, loss, pooling, position, fusion, or optimizer
paths. Retain the existing following `masked_fill`.

- [ ] **Step 3: Verify green and commit.**

Run focused model/config suites and commit:

```bash
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 availability context"
```

## Task 3: Runner Persistence, Check, And Strict Replay

- [ ] **Step 1: Write RED runner tests.**

Use a manually replaced fake `Q2Config` with
`temporal_context_variant="availability_embedding"`. Require both `check_q2`
and `run_q2` to accept it without the inaccessible Attachment 2 Test member;
the fake run must persist `training["temporal_context_variant"]`. Capture model
construction to require this exact value.

Create a strict saved-valid fixture with a context-model state dictionary and
persisted `availability_embedding`; capture construction and require the value
before archive access. Duplicate it with `"unsupported"` and an archive
sentinel that raises on all access; require the exact enum error and zero
sentinel calls. Finally remove the field from a valid historical manifest and
require reconstruction with `none` before valid-only evaluation.

Run `tests/test_q2_runner.py` focused nodes. Expected RED: manual configs are
not validated/passed, manifests omit the field, or saved replay defaults or
accepts incorrectly.

- [ ] **Step 2: Implement the persisted call chain.**

In `runner.py`, import the two context validators. Call
`validate_temporal_context_training` in `run_q2` and `check_q2` before output
or archive access; pass `config.temporal_context_variant` into model
construction. Write it beside `temporal_position_variant` in the manifest.

For strict replay add:

```python
def _manifest_temporal_context_variant(training: Mapping[str, object]) -> str:
    if "temporal_context_variant" not in training:
        return "none"
    return validate_temporal_context_variant(training["temporal_context_variant"])
```

Resolve it and validate its fusion compatibility before building or verifying
the archive, then pass it into strict model construction. Do not alter saved
inference math or access Attachment 2 Test.

- [ ] **Step 3: Verify all green and commit.**

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python \
  -m pytest tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py tests/test_cli.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
```

Commit runner/tests only after all commands pass:

```bash
git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py tests/test_cli.py
git commit -m "test: persist Q2 availability context"
```

## Task 4: One Valid-Only Run And Frozen Record

- [ ] **Step 1: Create and preflight an ignored A-normalized TOML.**

Copy every A setting exactly, use output
`artifacts/q2-valid-availability-embedding`, and change only
`temporal_context_variant = "availability_embedding"`. Run one:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python \
  -m e_mosei_audit.cli train-q2 --config q2-availability-embedding.toml --check
```

Require `train_count=3395`, `valid_count=728`, and `attachment3_count=30` with
no output directory. Do not invoke a Test command.

- [ ] **Step 2: Train and strictly reconstruct exactly once each.**

Run the same command once without `--check`, then run once with a fresh strict
report path:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python \
  -m e_mosei_audit.cli evaluate-q2-valid \
  --run-dir artifacts/q2-valid-availability-embedding \
  --output artifacts/q2-valid-availability-embedding/strict-valid-report.json
```

Require saved/rebuilt four clean metrics to match, support 728, 27 scenario
rows, 30 Attachment 3 predictions, a nonempty model, and a normalized A/T
manifest difference only in `temporal_context_variant`.

- [ ] **Step 3: Record and freeze.**

Append exactly one paired README Val/Test row. The Val row contains actual
metrics and the Test row is all `-` with `未评估`. Add a comparison JSON and
portfolio entry with metrics, class F1, scenario summary, strict equality,
counts, sole manifest difference, and train/valid-only scope. If macro-F1 is
below `0.6212527658`, retire Candidate T without any follow-up width,
initialization, bias, scale, placement, optimizer, checkpoint, or seed run.
