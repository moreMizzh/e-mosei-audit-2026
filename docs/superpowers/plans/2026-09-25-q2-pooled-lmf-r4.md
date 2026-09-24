# Q2 Pooled LMF-r4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exactly one `pooled_lmf_r4` Q2 fusion variant and evaluate it once from A using Attachment 2 train/valid only, accepting only clean valid macro-F1 `>= 0.6212527658`.

**Architecture:** Preserve A's masked gated sequence fusion, temporal encoder and pooler. Pool each projected modality only over independently available non-padding slots, apply three bias-free rank-4 factor groups, multiply their rank-wise outputs, and add the residual to A's pooled state only when all three modalities have usable frames. Rank four is in the variant name, never a config field.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers, pytest, local 7-Zip and local BERT. Use `/home/administrator/MyItem/E/.tools/q1-kaggle/bin/python`: ignored `.tools/` is intentionally absent from this linked worktree.

---

### Task 1: Register the Immutable Variant

**Files:**
- Modify: `src/e_mosei_audit/q2/config.py`
- Modify: `tests/test_q2_config.py`

- [ ] **Step 1: Write the failing config test.**

Add `"pooled_lmf_r4"` to the existing supported-fusion parametrization and the exact unsupported-fusion expectation in `tests/test_q2_config.py`. Do not add a rank TOML key.

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q
```

Expected RED: loading the new value raises the old unsupported fusion error.

- [ ] **Step 2: Permit exactly the public name.**

Append one item to `FUSION_VARIANTS` and its validation error in `src/e_mosei_audit/q2/config.py`:

```python
FUSION_VARIANTS = (
    "gated", "mag_lite", "mult_lite", "late_expert_shared",
    "text_anchor_residual", "pairwise_hadamard_residual", "pooled_lmf_r4",
)
```

No Q2Config field, default, parameter group, rank setting or hyperparameter search is allowed.

- [ ] **Step 3: Verify GREEN and commit.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_config.py -q
git diff --check
git add src/e_mosei_audit/q2/config.py tests/test_q2_config.py
git commit -m "feat: configure Q2 pooled LMF fusion"
```

### Task 2: Write LMF Algebra and Safety Tests First

**Files:**
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Add failing pure-helper tests.**

Import future `_POOLED_LMF_RANK` and `_pooled_lmf_residual`. Use `factors = torch.eye(2).repeat(3, 4, 1, 1)` with:

```python
states = (
    torch.tensor([[[2.0, 4.0], [4.0, 8.0], [101.0, 103.0]]]),
    torch.tensor([[[3.0, 5.0], [107.0, 109.0], [113.0, 127.0]]]),
    torch.tensor([[[7.0, 11.0], [13.0, 17.0], [131.0, 137.0]]]),
)
availability = torch.tensor([[[True, True, True], [True, False, True], [True, False, True]]])
temporal = torch.tensor([[True, True, False]])
expected = torch.tensor([[360.0, 1680.0]])
```

The expected value is `4 * ([3, 6] * [3, 5] * [10, 14])`. Add a second test with one modality having no valid position that requires an exactly zero residual.

- [ ] **Step 2: Add failing model-boundary tests.**

Construct `gated` and `pooled_lmf_r4` models with `hidden_size=16`, `heads=4`, `layers=1`, `dropout=0.0`; reset the same seed for each and copy non-LMF state entries from A to L. Add tests for:

```python
# 1. One or two globally available modalities: exact A fallback.
torch.testing.assert_close(lmf.logits, gated.logits, rtol=0.0, atol=0.0)
torch.testing.assert_close(lmf.score, gated.score, rtol=0.0, atol=0.0)
(lmf.logits.square().mean() + lmf.score.square().mean()).backward()
assert torch.count_nonzero(lmf_model.pooled_lmf_factors.grad) == 0

# 2. All three modalities available: L changes a prediction tensor.
assert not torch.equal(lmf.logits, gated.logits)

# 3. Large finite values at every unavailable raw position change no output.
# 4. Appended temporal padding changes neither helper residual nor outputs.
# 5. Added parameter count is exactly 3 * 4 * 16 * 16.
```

Also require finite `[B,3]` logits and `[B]` score, factor shape `(3, 4, 16, 16)`, and bit-identical A base parameters at equal initialization seed.

- [ ] **Step 3: Verify RED.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
```

Expected RED: collection fails because the constant/helper does not yet exist; after registration, model behavior and parameter tests still fail until the implementation is added.

### Task 3: Implement the Fixed-rank Pooled Residual

**Files:**
- Modify: `src/e_mosei_audit/q2/model.py`
- Modify: `tests/test_q2_model.py`

- [ ] **Step 1: Add LMF parameters after all existing A modules.**

Define `_POOLED_LMF_RANK = 4` at module scope. At the end of `MaskAwareTemporalFusion.__init__`, after every existing module and optional text adapter is initialized, add:

```python
if self.fusion_variant == "pooled_lmf_r4":
    self.pooled_lmf_factors = nn.Parameter(
        torch.empty(3, _POOLED_LMF_RANK, hidden_size, hidden_size)
    )
    for modality_factors in self.pooled_lmf_factors:
        for factor in modality_factors:
            nn.init.xavier_uniform_(factor)
```

This creates exactly `3 * 4 * D * D` weights, no biases or tunable scale.

- [ ] **Step 2: Add the pure helper after `_projection`.**

```python
def _pooled_lmf_residual(states, availability, temporal_mask, factors):
    masks = torch.stack(
        tuple(temporal_mask & availability[..., index] for index in range(3)), dim=1
    )
    counts = masks.sum(dim=2)
    pooled = torch.stack(
        tuple((state * masks[:, index].unsqueeze(-1)).sum(dim=1)
              for index, state in enumerate(states)),
        dim=1,
    )
    pooled = pooled / counts.clamp_min(1).to(dtype=pooled.dtype).unsqueeze(-1)
    projected = torch.einsum("bmd,mrdh->bmrh", pooled, factors)
    residual = (projected[:, 0] * projected[:, 1] * projected[:, 2]).sum(dim=1)
    usable = counts.gt(0).all(dim=1, keepdim=True)
    return torch.where(usable, residual, torch.zeros_like(residual))
```

Before computing, validate `factors.shape == (3, _POOLED_LMF_RANK, D, D)` and raise `ValueError` otherwise. The helper uses only masks, never numeric zero detection.

- [ ] **Step 3: Add the residual only after A pooling.**

Immediately after the existing temporal attention produces `pooled`, before concatenating availability fractions, add:

```python
if self.fusion_variant == "pooled_lmf_r4":
    pooled = pooled + _pooled_lmf_residual(
        states, availability, masks.temporal, self.pooled_lmf_factors
    )
```

Do not alter gate values, temporal masks, the encoder, pool attention, heads, losses, optimizer or other fusion branches.

- [ ] **Step 4: Verify GREEN and commit.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q
git diff --check
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 pooled LMF fusion"
```

### Task 4: Persist the Rank and Guard the Valid-only Runner

**Files:**
- Modify: `src/e_mosei_audit/q2/runner.py`
- Modify: `tests/test_q2_runner.py`

- [ ] **Step 1: Add a failing fake-archive train/valid test.**

Reuse `TrainValidPayloadWithInaccessibleTest` with `fusion_variant="pooled_lmf_r4"`. Its inaccessible `payload["test"]` and `.get("test")` sentinels must remain active. Require:

```python
assert summary["attachment3_count"] == 30
assert len(predictions) == 30
assert manifest["training"]["fusion_variant"] == "pooled_lmf_r4"
assert manifest["architecture"] == {"pooled_lmf_rank": 4}
assert archive.verify_count == 1
```

Add a saved-valid reconstruction test: save an L model, include `fusion_variant="pooled_lmf_r4"` in its manifest, spy on construction and `load_state_dict`, then require `observed_variants == ["pooled_lmf_r4"]`, `observed_strict == [True]`, and `report["sample_count"] == 3`.

Run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
```

Expected RED: the generated run manifest lacks the required architecture record. The strict-reconstruction test remains a permanent guard against silent A fallback.

- [ ] **Step 2: Add one conditional manifest entry.**

In `_write_run_outputs`, construct the existing manifest mapping before serializing it. When and only when `config.fusion_variant == "pooled_lmf_r4"`, add:

```python
manifest["architecture"] = {"pooled_lmf_rank": 4}
```

Then pass that mapping to `_write_json`. Do not add a config field and do not require the optional mapping when loading historical checkpoints; strict reconstruction uses the immutable variant name.

- [ ] **Step 3: Verify runner and full suite, then commit.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
git commit -m "test: cover Q2 pooled LMF valid-only run"
```

### Task 5: Preflight, Run L Once, and Record the Actual Result

**Files:**
- Create: `q2-pooled-lmf-r4.toml` (ignored local config)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-pooled-lmf-r4/` (ignored real artifact)
- Create: `/home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-pooled-lmf-r4.json` (ignored real comparison)
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md`

- [ ] **Step 1: Create the exact ignored configuration and prove it is fresh.**

Use A's resolved fields verbatim and change only `fusion_variant`:

```toml
[paths]
archive = "/home/administrator/MyItem/E/E题数据 (2).zip"
seven_zip = "/home/administrator/MyItem/E/.tools/bin/7za"
bert_model = "/home/administrator/MyItem/E/.tools/models/bert-base-uncased"
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-pooled-lmf-r4"

[training]
seed = 20260924
epochs = 30
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001
hidden_size = 128
heads = 4
layers = 2
dropout = 0.1
regression_loss_weight = 0.5
polarity_consistency_loss_weight = 0.0
class_weight_exponent = 1.0
synthetic_missingness_enabled = false
fusion_variant = "pooled_lmf_r4"
text_adapter_variant = "identity"
classification_variant = "flat"
device = "cuda"
```

Before preflight, run `git check-ignore -q q2-pooled-lmf-r4.toml`, `test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-pooled-lmf-r4`, and `test ! -e /home/administrator/MyItem/E/artifacts/q2-valid-comparison-no-train-missingness-pooled-lmf-r4.json`.

- [ ] **Step 2: Run only the read-only preflight.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-pooled-lmf-r4.toml --check
```

Require exactly `train_count=3395`, `valid_count=728`, `attachment3_count=30`; do not proceed otherwise.

- [ ] **Step 3: Execute exactly one real training run.**

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 \
  --config q2-pooled-lmf-r4.toml
```

Never rerun L, open test, use test-only evaluation, average seeds, change configuration or choose another checkpoint.

- [ ] **Step 4: Audit and record the decision.**

Require nonempty `model.pt`, 728 valid-report support, 27 scenario rows, 30 Attachment 3 predictions, `architecture.pooled_lmf_rank == 4`, equal normalizers, and a normalized training difference only of `fusion_variant: gated -> pooled_lmf_r4`. The comparison JSON must use actual values and include rank `4`, parameter increment `196608`, per-class F1, clean/scenario deltas, `728/27/30` counts and explicit valid-only scope.

Accept only if clean `macro_f1 >= 0.6212527658`. Otherwise append actual Val metrics, a same-model Test row marked `未评估`, and a permanent rejection record; do not change rank, factor initialization, weight decay, loss, learning rate, checkpoint or residual scale.

- [ ] **Step 5: Verify output reconstruction and commit docs.**

Run `evaluate-q2-valid` once with a new output path and require 728 samples plus four metrics equal to the saved artifact. Then run:

```bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
git status --short
git add README.md docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
git commit -m "docs: record Q2 pooled LMF valid result"
```
