# Q2 Frozen Text Output Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one frozen-BERT, mask-safe, Houlsby-inspired output bottleneck as an independently configurable single-variable candidate, then evaluate it once from the no-training-missingness A control on Attachment 2 train/valid only.

**Architecture:** `text_adapter_variant` is intentionally independent of `fusion_variant`: it changes the frozen BERT output before `text_projection`, whereas MAG/MulT change cross-modal fusion. `identity` is the default and legacy-manifest fallback. `houlsby_output_b32` applies one fixed `768 -> 32 -> ReLU -> 768` residual only at observed text positions, then re-masks before the existing projection. Existing modules are constructed first; adapter construction and initialization restore the CPU RNG state afterward, so identity and adapter have byte-identical common initialization and the same subsequent dropout RNG stream. This is a Houlsby-inspired output adapter, not a full Adapter-BERT reproduction.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers, pytest, local 7-Zip/BERT assets.

---

### Task 1: Add a Separate Text-Adapter Configuration Contract

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `docs/q2-config.example.toml`
- Modify: `tests/test_q2_config.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing config tests.**

Add `text_adapter_variant = "identity"` to every valid TOML fixture. Parameterize supported values over `"identity"` and `"houlsby_output_b32"`; an unsupported value must raise exactly `text_adapter_variant must be one of: identity, houlsby_output_b32`. Assert the normal loaded default fixture exposes `config.text_adapter_variant == "identity"`.

- [ ] **Step 2: Run config tests and confirm RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py tests/test_cli.py -q`.

Expected: valid fixtures fail for missing `text_adapter_variant`, and adapter values are not accepted by the absent parser.

- [ ] **Step 3: Implement the exact configuration boundary.**

In `config.py`, add:

```python
TEXT_ADAPTER_VARIANTS = ("identity", "houlsby_output_b32")
```

Insert `"text_adapter_variant"` into `_TRAINING_FIELDS`, add a `text_adapter_variant: str` field to `Q2Config`, validate the TOML value through:

```python
def validate_text_adapter_variant(value: object) -> str:
    if not isinstance(value, str) or value not in TEXT_ADAPTER_VARIANTS:
        raise ValueError("text_adapter_variant must be one of: identity, houlsby_output_b32")
    return value
```

Call that validator from `_validate_training` and when constructing `Q2Config`. Add `text_adapter_variant = "identity"` to the example TOML. Do not add bottleneck, activation, scale, or adapter dropout knobs.

- [ ] **Step 4: Verify GREEN and commit.**

Run the command from Step 2 and `git diff --check`. Commit only these four files as `feat: configure Q2 text output adapter`.

### Task 2: Implement the Mask-Safe Frozen Output Adapter

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Write failing model tests.**

Add tests with `text_adapter_variant="houlsby_output_b32"` that assert standard output shapes, finite outputs, and bounded score. With text availability false at selected positions and audio/vision still available, changing only raw BERT text values at every unavailable position from zero to `1_000_000` must leave logits, score, gates, and temporal attention exactly equal.

Add a gradient test with observed text that backpropagates `output.logits.square().sum() + output.score.square().sum()` and requires finite, nonzero gradients on both adapter Linear weight tensors. The frozen `FrozenBertEncoder` test remains unchanged: its model parameters have `requires_grad=False` and its encode call is inside `no_grad`.

Add a deterministic-initialization test:

```python
torch.manual_seed(41)
identity = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1, dropout=0.0)
identity_next = torch.rand(4)
torch.manual_seed(41)
adapter = MaskAwareTemporalFusion(
    hidden_size=16, heads=4, layers=1, dropout=0.0,
    text_adapter_variant="houlsby_output_b32",
)
adapter_next = torch.rand(4)
common = set(identity.state_dict()) & set(adapter.state_dict())
assert all(torch.equal(identity.state_dict()[name], adapter.state_dict()[name]) for name in common)
assert torch.equal(identity_next, adapter_next)
```

- [ ] **Step 2: Run model tests and confirm RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q`.

Expected: constructor rejects `text_adapter_variant`; new adapter tests cannot run.

- [ ] **Step 3: Implement the fixed adapter without perturbing common state.**

Extend `MaskAwareTemporalFusion.__init__` with `text_adapter_variant: str = "identity"` and validate it. Keep all existing modules, in their existing construction order, before adapter construction. Only for `houlsby_output_b32`, preserve/restore CPU RNG around the new modules and their deterministic near-identity initialization:

```python
rng_state = torch.get_rng_state()
try:
    self.text_adapter_down = nn.Linear(768, 32)
    self.text_adapter_up = nn.Linear(32, 768)
    nn.init.trunc_normal_(self.text_adapter_down.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
    nn.init.zeros_(self.text_adapter_down.bias)
    nn.init.trunc_normal_(self.text_adapter_up.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
    nn.init.zeros_(self.text_adapter_up.bias)
finally:
    torch.set_rng_state(rng_state)
```

Use `a=-0.02, b=0.02` for both truncated-normal calls; do not initialize the up projection to exact zero because that would block a gradient to the down projection on the first step. Do not register adapter modules for `identity`.

At the top of `forward`, before the existing `self.text_projection(text)` call and only for `houlsby_output_b32`, execute:

```python
text_mask = masks.text.unsqueeze(-1)
masked_text = text.masked_fill(~text_mask, 0.0)
delta = self.text_adapter_up(torch.relu(self.text_adapter_down(masked_text)))
delta = delta.masked_fill(~text_mask, 0.0)
text = (masked_text + delta).masked_fill(~text_mask, 0.0)
```

Then leave the existing projections, post-projection mask-zeroing, fusion variants, gate, temporal encoder, heads, and losses unchanged. The identity path must pass the raw text to the existing projection exactly as before.

- [ ] **Step 4: Verify GREEN and commit.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q`, then `git diff --check`. Commit only the model and test files as `feat: add Q2 frozen text output adapter`.

### Task 3: Persist and Restore the Adapter Variant

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Write failing runner tests.**

Update `runner_config` with `text_adapter_variant="identity"`. Add a fake-archive one-epoch run with `fusion_variant="gated"` and `text_adapter_variant="houlsby_output_b32"`; assert the manifest contains that value. Add a saved-validation test that saves a `houlsby_output_b32` state dict with a matching manifest and loads it strictly through `evaluate_saved_q2_valid`. Extend the existing old-manifest case without `fusion_variant` so it also omits `text_adapter_variant`; it must reconstruct `gated` plus `identity`. Add an invalid-manifest adapter value test that raises the exact configuration error. Use only the existing fake archive train/valid payload and inaccessible fake test sentinel.

- [ ] **Step 2: Run runner tests and confirm RED.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q`.

Expected: runner does not accept the new config field, does not write the manifest value, and saved validation cannot reconstruct the adapter checkpoint.

- [ ] **Step 3: Wire the value through training and saved evaluation.**

Pass `config.text_adapter_variant` to `MaskAwareTemporalFusion` in `run_q2`. Persist `"text_adapter_variant": config.text_adapter_variant` in `run_manifest.json`. In saved evaluation add:

```python
def _manifest_text_adapter_variant(training: Mapping[str, object]) -> str:
    if "text_adapter_variant" not in training:
        return "identity"
    return validate_text_adapter_variant(training["text_adapter_variant"])
```

Pass the helper result when rebuilding `MaskAwareTemporalFusion`. Do not change archive/member loading: both training and saved evaluation remain attached to `load_aligned_train_valid` only.

- [ ] **Step 4: Verify GREEN, full regression suite, and commit.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q`, then `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`, then `git diff --check`. Commit only runner/test files as `feat: persist Q2 text output adapter`.

### Task 4: Run and Audit the One Adapter Treatment

**Files:**
- Modify (ignored): `q2.toml`
- Create (ignored): `artifacts/q2-valid-text-adapter-b32/`
- Create (ignored): `artifacts/q2-valid-comparison-no-train-missingness-text-adapter-b32.json`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Set the exact one-variable configuration.**

Preserve A values: seed `20260924`, 30 epochs, batch size 64, learning rate `0.001`, weight decay `0.0001`, hidden size 128, four heads, two layers, dropout `0.1`, regression weight `0.5`, consistency weight `0.0`, class-weight exponent `1.0`, `synthetic_missingness_enabled=false`, and `fusion_variant="gated"`. Set only:

```toml
text_adapter_variant = "houlsby_output_b32"
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-text-adapter-b32"
```

- [ ] **Step 2: Preflight and train once.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml --check`, requiring exact `train=3395`, `valid=728`, and `attachment3=30`. Then run the same command without `--check` once. Never load, index, evaluate, or report Attachment 2 test.

- [ ] **Step 3: Audit and decide.**

Require a nonempty model, manifest, valid class support sum 728, 27 scenario rows, and 30 Attachment 3 predictions. Normalize A legacy defaults `polarity_consistency_loss_weight=0.0`, `fusion_variant="gated"`, and `text_adapter_variant="identity"`; every other compared training setting must match.

Accept only clean valid macro-F1 `>= 0.6212527658`, MAE no greater than v4 plus `0.02`, and scenario mean/worst macro-F1 each no more than `0.01` below A (`0.6037955229`, `0.5463436545`). Otherwise record exact valid-only metrics/deltas in the ignored comparison JSON, append the candidate to both README tables, update the exploration portfolio, and proceed to late-expert fusion without adapter bottleneck/activation/initialization tuning.

- [ ] **Step 4: Run full tests and commit the result record.**

Run `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q` and `git diff --check`. Commit the tracked documentation result only as `docs: record Q2 text adapter result`; ignored artifacts/configuration remain untracked.
