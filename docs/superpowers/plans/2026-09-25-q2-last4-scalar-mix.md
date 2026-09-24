# Q2 Frozen Last-Four-Layer Scalar-Mix Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate exactly one Q2 valid-only candidate that replaces frozen BERT's final layer with a five-parameter scalar mix of its last four frozen layers.

**Architecture:** Fresh configurations explicitly select `text_encoder_variant`; default `last_hidden_state` preserves historical final-layer behavior. `last4_scalar_mix` keeps BERT in eval/no-grad, learns only four softmax layer logits and one scalar scale outside BERT, and returns the unchanged `[B, 50, 768]` interface. The runner optimizes, snapshots, saves, and strictly restores that tiny encoder state independently from the strict fusion-model state.

**Tech Stack:** Python 3.12, PyTorch, Transformers, pytest, TOML, existing `e_mosei_audit.q2` runner, local offline BERT.

---

### Task 1: Persist the text-encoder representation variant

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_q2_runner.py`

- [x] **Step 1: Write failing configuration and fixture tests**

Add `text_encoder_variant = "last_hidden_state"` to every valid inline Q2 TOML and direct `Q2Config` fixture. Add a parameterized parser test accepting exactly `last_hidden_state` and `last4_scalar_mix`; add missing-field coverage requiring:

    missing required training field: text_encoder_variant

Add invalid-string and non-string tests requiring:

    text_encoder_variant must be one of: last_hidden_state, last4_scalar_mix

Update shared CLI and runner `Q2Config` construction to name `last_hidden_state` explicitly.

- [x] **Step 2: Verify the focused consumers are red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py -q

Expected: fresh valid TOMLs reject the unexpected field and direct configs fail because the dataclass does not accept it.

- [x] **Step 3: Implement the strict configuration contract**

In `config.py`, add:

```python
TEXT_ENCODER_VARIANTS = ("last_hidden_state", "last4_scalar_mix")

def validate_text_encoder_variant(value: object) -> str:
    if not isinstance(value, str) or value not in TEXT_ENCODER_VARIANTS:
        raise ValueError("text_encoder_variant must be one of: last_hidden_state, last4_scalar_mix")
    return value
```

Insert `text_encoder_variant` after `temporal_pooling_variant` in `_TRAINING_FIELDS`, add the frozen `Q2Config.text_encoder_variant: str`, validate/store it in `load_q2_config`, and add `text_encoder_variant = "last_hidden_state"` to the example TOML. Do not default a missing fresh configuration field.

- [x] **Step 4: Verify focused configuration consumers are green**

Run the Step 2 command. Expected: all focused configuration, CLI, and runner fixture users pass.

- [x] **Step 5: Commit Task 1**

    git add src/e_mosei_audit/q2/config.py docs/q2-config.example.toml tests/test_q2_config.py tests/test_cli.py tests/test_q2_runner.py
    git commit -m "feat: configure Q2 text encoder variant"

### Task 2: Add the frozen four-layer scalar mix

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `tests/test_q2_model.py`

- [x] **Step 1: Write failing frozen-encoder tests**

Extend `FakeBert` so it records `output_hidden_states` and can return a deterministic `hidden_states` tuple whose final four `[B, 50, 768]` tensors differ. Add tests requiring:

- default `FrozenBertEncoder(..., text_encoder_variant="last_hidden_state")` returns the unchanged final layer, never asks for hidden states, exposes no trainable parameters/state, and preserves BERT's frozen/no-grad behavior;
- `last4_scalar_mix` exposes exactly `layer_logits` shape `[4]` and scalar `scale`, initialized to zeros and one, respectively, with no successor RNG change;
- zero logits and unit scale produce the exact arithmetic mean of the final four supplied layers, retain `[B,50,768]`, and request hidden states only for this variant;
- a scalar loss on its output produces finite nonzero gradients for both `layer_logits` and `scale`, while every BERT parameter remains `requires_grad=False` and gradient-free;
- encoder trainable state contains exactly `layer_logits` and `scale`, accepts a matching detached state, and rejects missing, extra, non-tensor, non-finite, or wrong-shaped state without serializing BERT weights.

- [x] **Step 2: Run model tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q

Expected: construction rejects `text_encoder_variant`, and scalar-mix state/hidden-layer assertions fail because the interface does not exist.

- [x] **Step 3: Implement the minimal frozen encoder interface**

Import `validate_text_encoder_variant`. Extend `FrozenBertEncoder.__init__` and `from_local` with `text_encoder_variant: str = "last_hidden_state"`; validate and store it. Keep `self.model.eval()` and `self.model.requires_grad_(False)` unchanged.

For `last4_scalar_mix` only, after the frozen BERT model is installed, create exactly:

```python
self.layer_logits = nn.Parameter(torch.zeros(4, device=device))
self.scale = nn.Parameter(torch.ones((), device=device))
```

No random initialization is allowed. Add explicit methods returning the tuple of trainable tensors, the detached CPU state mapping `{"layer_logits": ..., "scale": ...}`, and a strict state loader that checks exactly these finite tensor names/shapes before copying them under `torch.no_grad()`.

In `encode`, call BERT under `torch.no_grad()` as today. The default path must make the existing call and return `output.last_hidden_state` unchanged. The scalar-mix path calls BERT with `output_hidden_states=True`, validates at least four tensor hidden states each shaped `[B,50,768]`, stacks `output.hidden_states[-4:]`, then computes:

```python
weights = torch.softmax(self.layer_logits, dim=0).view(4, 1, 1, 1)
embeddings = self.scale * (weights * layers).sum(dim=0)
```

Validate final shape/finite tensor as the existing encoder does. Do not register BERT in the fusion model, unfreeze it, add a b32 adapter, or change `MaskAwareTemporalFusion`.

- [x] **Step 4: Verify model tests are green**

Run the Step 2 command. Expected: all encoder and mask/fusion regression tests pass.

- [x] **Step 5: Commit Task 2**

    git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
    git commit -m "feat: add Q2 frozen last-four scalar mix"

### Task 3: Couple scalar-mix state to training and saved-valid reconstruction

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `tests/test_q2_runner.py`

- [x] **Step 1: Write failing runner tests**

Add a fake scalar-mix token encoder exposing `trainable_parameters`, `trainable_state_dict`, and strict state loading alongside the existing `encode` method. With an `last4_scalar_mix` config and inaccessible fake `test` member, require a one-epoch `run_q2` to:

- pass the selected variant to local encoder construction when no injected encoder is supplied;
- add the encoder's five trainable values to AdamW only for the scalar-mix variant;
- snapshot the encoder state with the same best epoch as the model, write `text_encoder_state.pt`, record `training.text_encoder_variant`, produce 30 Attachment 3 rows, and leave the forbidden test member unread.

Add saved-valid tests requiring scalar-mix manifest reconstruction to instantiate/select `last4_scalar_mix`, load `model.pt` with `strict=True`, strictly load `text_encoder_state.pt`, and report valid samples. Add historical-manifest coverage that omits `text_encoder_variant` and reconstructs `last_hidden_state` without requiring an encoder-state file. Add deterministic failures for an invalid manifest variant, a missing scalar-mix state file, and malformed scalar-mix state.

- [x] **Step 2: Run runner tests to prove red**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q

Expected: normal construction omits the variant, optimizer ignores encoder state, no scalar-mix state is written, or saved-valid cannot restore it.

- [x] **Step 3: Implement runner propagation, atomic persistence, and legacy fallback**

Add `_manifest_text_encoder_variant(training)` that returns `"last_hidden_state"` only if the field is absent and otherwise calls `validate_text_encoder_variant`. Pass `config.text_encoder_variant` to `FrozenBertEncoder.from_local` in `run_q2` and `check_q2`; pass the manifest variant in `evaluate_saved_q2_valid`.

Add narrow helpers that treat ordinary injected token encoders as having no trainable state, but require the explicit scalar-mix trainable/state interface when `last4_scalar_mix` is selected. Build AdamW from model parameters followed by encoder trainable parameters. Each time `_is_better` accepts an epoch, clone both `model.state_dict()` and the encoder trainable state to CPU. Restore both accepted states before valid report, scenario, and Attachment 3 prediction.

Extend `_write_run_outputs` to receive the selected encoder state and variant. Record `text_encoder_variant` in `manifest["training"]`; for `last4_scalar_mix`, atomically write `text_encoder_state.pt` with only the two detached tensors before publishing. Do not create that file for the default variant.

During saved-valid evaluation, preserve `model.load_state_dict(..., strict=True)`. After creating the local scalar-mix encoder, require/load `text_encoder_state.pt` through its strict loader before `_evaluate`; default/historical final-layer runs must not require the file. Do not load Attachment 3 or the Attachment 2 test split in this command.

- [x] **Step 4: Verify runner and CLI tests are green**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py tests/test_cli.py -q

Expected: runner and CLI tests pass, fake test sentinels remain unread, and saved-valid remains strict.

- [x] **Step 5: Commit Task 3**

    git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
    git commit -m "feat: persist Q2 scalar-mix encoder state"

### Task 4: Run candidate O once and record the outcome

**Files:**
- Create ignored: `/home/administrator/MyItem/E/q2-last4-scalar-mix.toml`
- Create ignored: `artifacts/q2-valid-last4-scalar-mix/`
- Create ignored: `artifacts/q2-valid-comparison-no-train-missingness-last4-scalar-mix.json`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`
- Modify: `docs/superpowers/plans/2026-09-25-q2-last4-scalar-mix.md`

- [x] **Step 1: Run the full regression gate**

Run:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
    git diff --check
    test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix

Expected: all non-opt-in tests pass, no whitespace errors, and the target output directory does not exist.

- [x] **Step 2: Create the one-run A-derived configuration**

Create the ignored `/home/administrator/MyItem/E/q2-last4-scalar-mix.toml` with A's archive, 7-Zip, local BERT paths, and every A training value. Set only:

```toml
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix"
text_encoder_variant = "last4_scalar_mix"
```

Retain `seed=20260924`, `fusion_variant="gated"`, `text_adapter_variant="identity"`, `classification_variant="flat"`, `temporal_position_variant="none"`, `temporal_pooling_variant="attention"`, and `synthetic_missingness_enabled=false`.

- [x] **Step 3: Prove preflight and output absence**

Run:

    test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-last4-scalar-mix.toml --check
    test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix

Expected JSON is `{"attachment3_count": 30, "train_count": 3395, "valid_count": 728}` and no output directory.

- [x] **Step 4: Run exactly once and strictly reconstruct valid**

Run exactly once:

    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config /home/administrator/MyItem/E/q2-last4-scalar-mix.toml
    PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli evaluate-q2-valid --run-dir /home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix --output /home/administrator/MyItem/E/artifacts/q2-valid-last4-scalar-mix/recomputed_valid.json

Require exact saved/recomputed metrics, valid report support total 728, 27 scenario rows, 30 Attachment 3 predictions, nonempty `model.pt`, an encoder state with exactly `layer_logits` and `scale`, and manifest `text_encoder_variant="last4_scalar_mix"`. Do not run or create a test evaluation.

- [x] **Step 5: Record the acceptance decision without test**

Create the ignored A/O comparison JSON with clean metrics, per-class F1, scenario mean/worst, metric deltas, architecture `{last_four_layers: true, trainable_parameter_increment: 5}`, exact artifact checks, normalized manifest difference, and explicit train/valid-only scope. Update the top README Val row and a Test `未评估` row plus the exploration portfolio. Accept only clean valid macro-F1 `>= 0.6212527658`; otherwise retire this exact four-layer, zero-logit, unit-scale mix without tuning layer count, initialization, scale parameterization, BERT freeze/mode, optimizer, loss, checkpoint, seed, or b32 combination.

- [x] **Step 6: Commit verified documentation**

    git add README.md docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md docs/superpowers/plans/2026-09-25-q2-last4-scalar-mix.md
    git commit -m "docs: record Q2 last-four scalar mix result"

## Plan Review

Task 1 makes the fresh configuration self-describing. Task 2 isolates BERT-free scalar mixing and proves default preservation. Task 3 prevents optimizer/checkpoint mismatch by treating encoder state as an explicit two-tensor artifact with legacy fallback. Task 4 enforces one valid-only run, strict reconstruction, artifacts counts, and a no-test decision. The plan never changes a retired candidate's knobs or introduces a sweep.
