# Q2 Depthwise Local Residual Candidate U Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement and evaluate one A-derived valid-only Candidate U whose sole normalized training change is a zero-initialized depthwise local temporal residual.

**Architecture:** Add temporal_residual_variant with none and depthwise_conv3. The selected branch adds only a zero-initialized Conv1d(H,H,kernel_size=3,padding=1,groups=H,bias=False) to the already-masked gated fused sequence and reapplies the temporal mask before the existing Transformer. The runner persists and strictly reconstructs this semantic.

**Tech Stack:** Python 3.12, PyTorch, pytest, TOML, the current Q2 package, local offline BERT, and explicit local 7-Zip.

---

## File Map

- docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md: reconcile historical Candidate Q, then record U after the one run.
- src/e_mosei_audit/q2/config.py: residual enum, required field, and early late-expert compatibility check.
- src/e_mosei_audit/q2/model.py: zero-initialized, mask-safe convolution.
- src/e_mosei_audit/q2/runner.py: validation, construction, manifest, and strict valid-replay fallback.
- docs/q2-config.example.toml: explicit none default.
- tests/test_q2_config.py, tests/test_q2_model.py, tests/test_q2_runner.py, tests/test_cli.py: persisted contract and Test isolation.
- README.md: one paired U Val/Test row after the one run.
- Ignored q2-depthwise-local-residual.toml and artifacts/: a preflight, one run, one strict replay, and comparison record.

### Task 1: Repair The Historical Candidate Ledger

**Files:**
- Modify: docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
- Inspect: artifacts/q2-valid-comparison-no-train-missingness-neutral-gate-polarity.json

- [ ] **Step 1: Add the omitted Candidate Q result.**

Append Q after P and before R in the stage record and candidate table. Record that classification_variant="neutral_gate_polarity" produced clean macro-F1 0.5967240265878949, scenario mean/worst 0.5943445661782981/0.5500329163923635, and no Test evaluation. State that Q is retired and only archival: current code accepts flat and corn, so Q is not a current strict-replay contract. Do not restore or reuse Q.

- [ ] **Step 2: Check and commit the ledger repair.**

Run:

~~~bash
git diff --check
rg -n 'Candidate Q|neutral_gate_polarity|0\.5967240265878949'   docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
git add docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
git commit -m "docs: record retired Q2 neutral gate candidate"
~~~

### Task 2: Persist The Local-Residual Contract

**Files:**
- Modify: src/e_mosei_audit/q2/config.py
- Modify: docs/q2-config.example.toml
- Modify: tests/test_q2_config.py, tests/test_q2_runner.py, tests/test_cli.py

- [ ] **Step 1: Write RED config tests and complete all fixtures.**

Insert temporal_residual_variant = "none" after temporal_context_variant in every complete inline TOML fixture and direct Q2Config constructor.  Add this helper beside `_write_temporal_context_config`:

~~~python
def _write_temporal_residual_config(
    tmp_path: Path,
    *,
    temporal_residual_variant: str | None = '"none"',
    fusion_variant: str = "gated",
) -> Path:
    config_path = _write_temporal_context_config(tmp_path, fusion_variant=fusion_variant)
    residual_line = (
        "" if temporal_residual_variant is None
        else f"temporal_residual_variant = {temporal_residual_variant}\n"
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'temporal_pooling_variant = "attention"\n',
            f'{residual_line}temporal_pooling_variant = "attention"\n',
        ),
        encoding="utf-8",
    )
    return config_path

with pytest.raises(
    ValueError,
    match=r"\Adepthwise_conv3 temporal residual is unsupported with late_expert_shared fusion\Z",
):
    load_q2_config(
        _write_temporal_residual_config(
            tmp_path,
            temporal_residual_variant='"depthwise_conv3"',
            fusion_variant="late_expert_shared",
        )
    )
~~~

The missing and unknown errors must be:

~~~text
missing required training field: temporal_residual_variant
temporal_residual_variant must be one of: none, depthwise_conv3
~~~

Run before production code:

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m pytest tests/test_q2_config.py -q
~~~

Expected: RED because no residual field or validators exist.

- [ ] **Step 2: Implement the explicit config contract.**

In config.py, add:

~~~python
TEMPORAL_RESIDUAL_VARIANTS = ("none", "depthwise_conv3")

def validate_temporal_residual_variant(value: object) -> str:
    if not isinstance(value, str) or value not in TEMPORAL_RESIDUAL_VARIANTS:
        raise ValueError("temporal_residual_variant must be one of: none, depthwise_conv3")
    return value

def validate_temporal_residual_training(value: object, *, fusion_variant: str) -> str:
    variant = validate_temporal_residual_variant(value)
    if variant == "depthwise_conv3" and fusion_variant == "late_expert_shared":
        raise ValueError("depthwise_conv3 temporal residual is unsupported with late_expert_shared fusion")
    return variant
~~~

Add the required field after temporal_context_variant in _TRAINING_FIELDS and Q2Config. Validate it in _validate_training, parse it in load_q2_config, and add temporal_residual_variant = "none" to the example TOML. A fresh config has no silent default.

- [ ] **Step 3: Verify green and commit the config surface.**

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m pytest tests/test_q2_config.py tests/test_cli.py -q
git add src/e_mosei_audit/q2/config.py docs/q2-config.example.toml   tests/test_q2_config.py tests/test_q2_runner.py tests/test_cli.py
git commit -m "feat: configure Q2 local temporal residual"
~~~

### Task 3: Add The Mask-Safe Zero Residual

**Files:**
- Modify: src/e_mosei_audit/q2/model.py
- Modify: tests/test_q2_model.py

- [ ] **Step 1: Write RED model tests.**

Initialize matching none and depthwise_conv3 models from the same seed. Require the sole new key, shape, parameter count, common state, RNG successor, and public outputs to be exact:

~~~python
assert local.local_depthwise_residual.weight.shape == (16, 1, 3)
assert torch.equal(local.local_depthwise_residual.weight, torch.zeros(16, 1, 3))
assert set(local.state_dict()) - set(none.state_dict()) == {"local_depthwise_residual.weight"}
assert sum(p.numel() for p in local.parameters()) - sum(p.numel() for p in none.parameters()) == 48
assert torch.equal(local_successor, none_successor)
masks = availability_pooling_masks()
text = torch.randn(2, 5, 768)
audio = torch.randn(2, 5, 74)
vision = torch.randn(2, 5, 35)
assert_same_public_output(
    local(text=text, audio=audio, vision=vision, masks=masks),
    none(text=text, audio=audio, vision=vision, masks=masks),
)
~~~

Set the convolution weights nonzero and use RecordingEncoder to require unavailable raw values, an internal all-unavailable slot, and appended 1_000_000.0 padding to leave valid outputs unchanged. Its Transformer input must be zero at every invalid position. Backpropagate logits.square().sum() + score.square().sum() and require a finite, nonzero local weight gradient. Run focused model tests; expected RED is the missing option/state key.

- [ ] **Step 2: Implement only the frozen operator.**

Import the residual training validator and add temporal_residual_variant: str = "none" to the model constructor. After fusion compatibility is known, store its validated result. Construct the only new module while restoring Torch RNG:

~~~python
rng_state = torch.get_rng_state()
try:
    self.local_depthwise_residual = nn.Conv1d(
        hidden_size, hidden_size, kernel_size=3, stride=1, padding=1,
        dilation=1, groups=hidden_size, bias=False,
    )
    nn.init.zeros_(self.local_depthwise_residual.weight)
finally:
    torch.set_rng_state(rng_state)
~~~

After the existing gated sum, availability-context branch, and first temporal masked_fill, add only:

~~~python
if self.temporal_residual_variant == "depthwise_conv3":
    local = self.local_depthwise_residual(fused.transpose(1, 2)).transpose(1, 2)
    fused = (fused + local).masked_fill(~temporal.unsqueeze(-1), 0.0)
~~~

Do not modify late-expert, pooling, position, fusion, loss, optimizer, or BERT paths. The first mask must happen before convolution.

- [ ] **Step 3: Verify green and commit the model.**

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m pytest tests/test_q2_model.py tests/test_q2_config.py -q
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py
git commit -m "feat: add Q2 depthwise local residual"
~~~

### Task 4: Persist U Through Training And Strict Valid Replay

**Files:**
- Modify: src/e_mosei_audit/q2/runner.py
- Modify: tests/test_q2_runner.py

- [ ] **Step 1: Write RED runner and replay tests.**

Starting from runner_config, substitute temporal_residual_variant="depthwise_conv3". Require check_q2 and run_q2 to work with runner_archive_with_inaccessible_test(), persist the field, and forward it exactly once to the model. Parameterize unknown and late-expert U semantics in both entrypoints, require the exact error, zero ArchiveAccessSentinel accesses, and no output directory.

Create a saved U model/manifest fixture.  Strict valid replay must construct U
and call `load_state_dict(state_dict, strict=True)`.  A missing historical
field must construct none; unknown and late-expert manifest values must fail
before archive access.  Focused runner tests are RED until plumbing is added.

- [ ] **Step 2: Implement the persisted runner chain.**

Import residual validators. In run_q2 and check_q2, call this before _validate_output_target or archive construction:

~~~python
validate_temporal_residual_training(
    config.temporal_residual_variant,
    fusion_variant=config.fusion_variant,
)
~~~

Pass the field to every MaskAwareTemporalFusion construction and write it beside temporal_context_variant in the manifest. Add:

~~~python
def _manifest_temporal_residual_variant(training: Mapping[str, object]) -> str:
    if "temporal_residual_variant" not in training:
        return "none"
    return validate_temporal_residual_variant(training["temporal_residual_variant"])
~~~

Strict replay resolves and compatibility-validates it before archive verification, then passes it to the model. Keep strict state loading and do not open Attachment 2 Test.

- [ ] **Step 3: Verify all automated contracts and commit.**

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m pytest tests/test_q2_config.py tests/test_q2_model.py tests/test_q2_runner.py tests/test_cli.py -q
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q
git diff --check
git add src/e_mosei_audit/q2/runner.py tests/test_q2_runner.py
git commit -m "test: persist Q2 depthwise local residual"
~~~

### Task 5: Run U Once And Freeze The Real Result

**Files:**
- Create (ignored): q2-depthwise-local-residual.toml
- Create (ignored): artifacts/q2-valid-depthwise-local-residual/
- Create (ignored): artifacts/q2-valid-comparison-no-train-missingness-depthwise-local-residual.json
- Modify: README.md, docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md

- [ ] **Step 1: Create and preflight the exact A-derived TOML.**

Create this exact ignored configuration, changing no A field except
`output_dir` and `temporal_residual_variant`:

~~~toml
[paths]
archive = "/home/administrator/MyItem/E/E题数据 (2).zip"
seven_zip = "/home/administrator/MyItem/E/.tools/bin/7za"
bert_model = "/home/administrator/MyItem/E/.tools/models/bert-base-uncased"
output_dir = "/home/administrator/MyItem/E/artifacts/q2-valid-depthwise-local-residual"

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
dropout_consistency_variant = "none"
class_weight_exponent = 1.0
synthetic_missingness_enabled = false
fusion_variant = "gated"
text_adapter_variant = "identity"
classification_variant = "flat"
classification_loss_variant = "hard_ce"
temporal_position_variant = "none"
temporal_context_variant = "none"
temporal_residual_variant = "depthwise_conv3"
temporal_pooling_variant = "attention"
text_encoder_variant = "last_hidden_state"
device = "cuda"
~~~

Run exactly once:

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m e_mosei_audit.cli train-q2 --config q2-depthwise-local-residual.toml --check
~~~

Require train_count=3395, valid_count=728, and attachment3_count=30 without creating an output. Do not invoke a Test command.

- [ ] **Step 2: Train exactly once and strictly reconstruct exactly once.**

~~~bash
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m e_mosei_audit.cli train-q2 --config q2-depthwise-local-residual.toml
PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python   -m e_mosei_audit.cli evaluate-q2-valid   --run-dir /home/administrator/MyItem/E/artifacts/q2-valid-depthwise-local-residual   --output /home/administrator/MyItem/E/artifacts/q2-valid-depthwise-local-residual/strict-valid-report.json
~~~

Require equal saved/rebuilt clean metrics and confusion matrix, support 728, 27 scenario rows, 30 Attachment 3 predictions, a nonempty model.pt, and the sole normalized A/U manifest difference temporal_residual_variant: none -> depthwise_conv3.

- [ ] **Step 3: Record actual metrics and permanently decide U.**

Create the comparison JSON with the existing valid-only scope, clean metrics, class F1, scenario mean/worst, 728/27/30 counts, strict replay equality, model-state evidence, and sole manifest difference. Append one paired README row: actual U values in Val and all - plus 未评估 in Test. Append the portfolio result. If clean macro-F1 is below 0.6212527658, retire this exact kernel/padding/initialization/placement/bias/scale/optimizer/checkpoint/seed without a retry, sweep, or combination; otherwise promote the frozen U model.

~~~bash
git add README.md docs/superpowers/specs/2026-09-24-q2-exploration-portfolio.md
git commit -m "docs: record Q2 depthwise local residual result"
~~~
