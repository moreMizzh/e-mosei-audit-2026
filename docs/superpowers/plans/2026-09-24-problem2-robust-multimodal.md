# 问题 2：鲁棒多模态情感预测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用附件 2 对齐特征训练掩码感知的联合情感分类/回归模型，评估连续局部缺失鲁棒性，并生成附件 3 的 30 条最终预测。

**Architecture:** `q2.data` 负责只读归档输入和数据契约；`q2.missingness` 产生/识别连续块掩码；`q2.model` 在冻结 BERT 输出上执行可用性门控时序融合；`q2.runner` 训练、验证、写出原子化的可追溯产物。CLI 增加 `train-q2`，配置文件必须显式给出归档、7-Zip、本地 BERT 和新输出目录。

**Tech Stack:** Python 3.12、NumPy、PyTorch、Transformers、本项目 `SevenZipArchive`、pytest。

---

## 文件责任图

| 路径 | 责任 |
| --- | --- |
| `src/e_mosei_audit/q2/data.py` | 附件 2/3 流式读取、token 规范化、样本与训练统计量。 |
| `src/e_mosei_audit/q2/missingness.py` | 模态可用性和可复现连续块扰动。 |
| `src/e_mosei_audit/q2/model.py` | 冻结 BERT 文本编码、掩码门控融合、联合输出头。 |
| `src/e_mosei_audit/q2/config.py` | TOML 配置和路径/超参数预检。 |
| `src/e_mosei_audit/q2/runner.py` | 训练、指标、场景矩阵、附件 3 推理和事务输出。 |
| `src/e_mosei_audit/cli.py` | `train-q2` 子命令，不改变已有命令。 |
| `docs/q2-config.example.toml` | 唯一的可复制运行入口。 |
| `tests/test_q2_*.py` | 对应模块的纯单元测试及小 ZIP 端到端测试。 |

### Task 1: 文本与专项输入契约

**Files:**
- Create: `src/e_mosei_audit/q2/__init__.py`
- Create: `src/e_mosei_audit/q2/data.py`
- Create: `tests/test_q2_data.py`

- [ ] **Step 1: 写出共同 `text_bert` 输入的失败测试。**

```python
def test_normalise_text_bert_accepts_integral_float_attachment3_tokens() -> None:
    raw = np.zeros((1, 3, 50), dtype=np.float32)
    raw[0, 0, :2] = [101.0, 102.0]
    raw[0, 1, :2] = 1.0
    tokens = normalise_text_bert(raw)
    assert tokens.dtype == np.int64
    np.testing.assert_array_equal(tokens, raw.astype(np.int64))


def test_normalise_text_bert_rejects_non_integral_tokens() -> None:
    raw = np.zeros((1, 3, 50), dtype=np.float32)
    raw[0, 0, 1] = 101.5
    with pytest.raises(DataContractError, match="integer-valued"):
        normalise_text_bert(raw)
```

- [ ] **Step 2: 确认测试因缺少模块失败。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_data.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'e_mosei_audit.q2'`.

- [ ] **Step 3: 最小实现数据类型和形状验证。**

```python
def normalise_text_bert(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[1:] != (3, 50):
        raise DataContractError("text_bert must have shape [N, 3, 50]")
    if not np.isfinite(value).all() or not np.equal(value, np.rint(value)).all():
        raise DataContractError("text_bert must contain finite integer-valued tokens")
    return value.astype(np.int64, copy=False)
```

Define immutable `AlignedSplit` and `Attachment3Sample`; `load_aligned_dataset` must read only `aligned_50.pkl` from `SevenZipArchive.open_member`, require `audio[N,50,74]`/`vision[N,50,35]`, map class labels to `int64`, and reject labels outside `0,1,2`. `load_attachment3_aligned` must sort Chinese member filenames naturally, read one pickle at a time, and make file-stable IDs from `attachment3:aligned:附件3_01` through `attachment3:aligned:附件3_30`.

- [ ] **Step 4: 扩充失败测试为成员排序、字段缺失和标签映射。**

```python
def test_attachment3_ids_follow_natural_member_order(fake_archive: FakeArchive) -> None:
    samples = load_attachment3_aligned(fake_archive)
    assert [sample.sample_id for sample in samples] == [
        "attachment3:aligned:附件3_01", "attachment3:aligned:附件3_02"
    ]


def test_aligned_split_rejects_class_outside_three_polarities() -> None:
    with pytest.raises(DataContractError, match="classification_labels"):
        aligned_split_from_mapping({**valid_split(), "classification_labels": np.array([3.0])})
```

- [ ] **Step 5: 实现并运行本任务所有数据测试。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_data.py -q`

Expected: PASS.

- [ ] **Step 6: 提交数据契约。**

```bash
git add src/e_mosei_audit/q2/__init__.py src/e_mosei_audit/q2/data.py tests/test_q2_data.py
git commit -m "feat: add Q2 aligned data contracts"
```

### Task 2: 连续局部缺失与标准化

**Files:**
- Create: `src/e_mosei_audit/q2/missingness.py`
- Create: `tests/test_q2_missingness.py`
- Modify: `src/e_mosei_audit/q2/data.py`

- [ ] **Step 1: 写出基础可用性和不跨 padding 的失败测试。**

```python
def test_observed_masks_use_attention_and_zero_vector_evidence() -> None:
    text_bert = np.array([[[101, 102, 0], [1, 1, 0], [0, 0, 0]]])
    audio = np.ones((1, 3, 74)); audio[:, 1] = 0
    vision = np.ones((1, 3, 35)); vision[:, 2] = 0
    masks = observed_masks(text_bert, audio, vision)
    np.testing.assert_array_equal(masks.text, [[True, True, False]])
    np.testing.assert_array_equal(masks.audio, [[True, False, False]])
    np.testing.assert_array_equal(masks.temporal, [[True, True, False]])


def test_contiguous_drop_never_changes_padding_or_labels() -> None:
    result = apply_contiguous_drop(batch, rng=np.random.default_rng(7), modalities=("audio",))
    assert not result.audio_available[0, 3:].any()
    np.testing.assert_array_equal(result.labels, batch.labels)
```

- [ ] **Step 2: 运行并确认失败。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_missingness.py -q`

Expected: FAIL because mask functions are absent.

- [ ] **Step 3: 实现观测掩码、训练统计量和块扰动。**

`observed_masks` returns bool arrays for text/audio/vision/temporal. `fit_normalizer` consumes only training `audio`/`vision` positions where both temporal and modality masks are true, clamps standard deviations below `1e-6` to 1, and returns JSON-safe state. `apply_contiguous_drop` chooses a block entirely inside per-sample true temporal positions, creates `SyntheticDrop` provenance (`modalities`, `start`, `length`, `fraction`), and only zeros the copied model input/mask.

- [ ] **Step 4: 添加确定性场景矩阵失败测试。**

```python
def test_validation_scenarios_cover_every_modality_position_and_duration() -> None:
    scenarios = validation_scenarios()
    assert len(scenarios) == 27
    assert {(s.modality, s.position, s.fraction) for s in scenarios} == {
        (m, p, f) for m in ("text", "audio", "vision")
        for p in ("beginning", "middle", "end") for f in (0.1, 0.3, 0.5)
    }
```

- [ ] **Step 5: 实现场景构造并验证。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_missingness.py -q`

Expected: PASS.

- [ ] **Step 6: 提交。**

```bash
git add src/e_mosei_audit/q2/data.py src/e_mosei_audit/q2/missingness.py tests/test_q2_missingness.py
git commit -m "feat: add Q2 contiguous missingness controls"
```

### Task 3: 掩码感知联合模型

**Files:**
- Create: `src/e_mosei_audit/q2/model.py`
- Create: `tests/test_q2_model.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 写出模型输出范围和不可用模态门控的失败测试。**

```python
def test_mask_aware_fusion_returns_three_logits_and_bounded_score() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1)
    output = model(text=torch.randn(2, 50, 768), audio=torch.randn(2, 50, 74),
                   vision=torch.randn(2, 50, 35), masks=example_masks())
    assert output.logits.shape == (2, 3)
    assert output.score.shape == (2,)
    assert torch.all(output.score <= 3) and torch.all(output.score >= -3)


def test_gate_assigns_zero_weight_to_an_unavailable_modality() -> None:
    model = MaskAwareTemporalFusion(hidden_size=16, heads=4, layers=1)
    output = model(text=torch.ones(2, 50, 768), audio=torch.ones(2, 50, 74),
                   vision=torch.ones(2, 50, 35), masks=example_masks(audio=False))
    assert torch.equal(output.gates[:, :, 1], torch.zeros_like(output.gates[:, :, 1]))
```

- [ ] **Step 2: 运行并确认失败。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q`

Expected: FAIL because `MaskAwareTemporalFusion` does not exist.

- [ ] **Step 3: 实现可注入的文本编码器和核心融合模型。**

Define `FrozenBertEncoder(model_path, device)` with `BertModel.from_pretrained(str(model_path), local_files_only=True)`, `eval()`, `requires_grad_(False)`, and no-grad encoding of the three `text_bert` rows. Define core `MaskAwareTemporalFusion` separately so tests use direct 768-d text tensors. It projects each modality to one common hidden width, masks gate logits before softmax, zeros all-unavailable positions before Transformer, uses masked attention pooling, and emits `Q2Output(logits, score, gates, temporal_attention)`.

Add optional dependency group:

```toml
q2 = ["torch>=2.4", "transformers>=4.45"]
```

- [ ] **Step 4: 运行模型测试。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_model.py -q`

Expected: PASS.

- [ ] **Step 5: 提交。**

```bash
git add src/e_mosei_audit/q2/model.py tests/test_q2_model.py pyproject.toml
git commit -m "feat: add Q2 mask aware fusion model"
```

### Task 4: 配置、训练、受控验证和附件 3 预测

**Files:**
- Create: `src/e_mosei_audit/q2/config.py`
- Create: `src/e_mosei_audit/q2/runner.py`
- Modify: `src/e_mosei_audit/cli.py`
- Create: `docs/q2-config.example.toml`
- Create: `tests/test_q2_runner.py`
- Modify: `README.md`

- [ ] **Step 1: 写出事务输出和四项指标的失败测试。**

```python
def test_metric_summary_reports_accuracy_macro_f1_mae_and_pearson() -> None:
    metrics = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]),
                              np.array([-1., 0., 1.]), np.array([-1., 0., 1.]))
    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "mae": 0.0, "pearson": 1.0}


def test_prediction_csv_uses_fixed_polarity_mapping(tmp_path: Path) -> None:
    write_predictions(tmp_path, [Prediction("x", "f.pkl", 0, -0.5)])
    rows = list(csv.DictReader((tmp_path / "attachment3_predictions.csv").open()))
    assert rows[0]["polarity_label"] == "Negative"
```

- [ ] **Step 2: 运行并确认失败。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q`

Expected: FAIL because runner helpers do not exist.

- [ ] **Step 3: 实现配置和运行器。**

`Q2Config.from_toml` requires paths to the last `.zip` volume, executable 7-Zip, local BERT directory, and a nonexistent output directory. `run_q2` verifies archive before creating a staging directory, trains with cross-entropy plus `0.5 * smooth_l1`, selects epoch from clean validation `macro_f1` then lower MAE, emits clean metrics and all 27 scenarios, writes state/statistics/seeds into JSON, and atomically renames the staging directory only after 30 attachment-3 rows exist. Pearson returns `None` rather than a fabricated number for constant inputs.

Add `train-q2 --config PATH` alongside existing CLI subcommands. The example config uses repository-relative paths and `artifacts/q2-default`; README must state the exact command, required local BERT cache, metric boundary, and no-download/no-external-data guarantee.

- [ ] **Step 4: 添加小型 ZIP 端到端失败测试。**

```python
def test_run_q2_writes_30_attachment_predictions_for_aligned_members(tmp_path: Path) -> None:
    output = run_q2(tiny_config(tmp_path), token_encoder=TinyTokenEncoder())
    assert len(read_csv(output / "attachment3_predictions.csv")) == 30
    assert len(read_csv(output / "validation_scenarios.csv")) == 27
    assert json.loads((output / "metrics.json").read_text())["clean"]["accuracy"] is not None
```

The fixture must contain train/valid arrays and 30 in-memory attachment 3 pickle members; it must not use real competition data or a downloaded model.

- [ ] **Step 5: 最小实现后验证运行器和 CLI 帮助。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest tests/test_q2_runner.py -q`

Expected: PASS.

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --help`

Expected: exit 0 and displays `--config`.

- [ ] **Step 6: 提交。**

```bash
git add src/e_mosei_audit/q2/config.py src/e_mosei_audit/q2/runner.py src/e_mosei_audit/cli.py docs/q2-config.example.toml tests/test_q2_runner.py README.md
git commit -m "feat: add reproducible Q2 training workflow"
```

### Task 5: 全套验证和真实运行

**Files:**
- Modify: `README.md` only if verification finds a factual mismatch.

- [ ] **Step 1: 跑全部回归测试。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m pytest -q`

Expected: all default tests pass; only pre-existing opt-in real tests may be skipped.

- [ ] **Step 2: 检查配置和归档，不启动训练。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml --check`

Expected: integrity, interface, local-model and empty-output preflight succeed without writing an output directory.

- [ ] **Step 3: 启动一次确定性真实训练。**

Run: `PYTHONPATH="$PWD/src" /home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -m e_mosei_audit.cli train-q2 --config q2.toml`

Expected: exit 0; output has `metrics.json`, 27 validation scenario rows, 30 prediction rows, `model.pt`, and manifest.

- [ ] **Step 4: 独立核验提交边界。**

```bash
/home/administrator/MyItem/E/.tools/q1-kaggle/bin/python -c '
import csv, json
root = "artifacts/q2-default"
assert len(list(csv.DictReader(open(f"{root}/attachment3_predictions.csv")))) == 30
assert len(list(csv.DictReader(open(f"{root}/validation_scenarios.csv")))) == 27
metrics = json.load(open(f"{root}/metrics.json"))["clean"]
assert set(metrics) == {"accuracy", "macro_f1", "mae", "pearson"}
'
```

Expected: exit 0. Report validation metrics as validation-only and attachment 3 as unlabelled inference only.

- [ ] **Step 5: 提交实现与文档。**

```bash
git add README.md
git commit -m "docs: record Q2 validated workflow"
```

## 计划自检

- [ ] 数据共同接口、无外部情感数据、对齐版、训练增强、4 个验证指标、27 个受控场景、30 条附件 3 推理、可复现配置与产物均有明确任务。
- [ ] 所有实现任务均先给出具体失败测试、再给出通过命令。
- [ ] 不包含问题 1/3 改动，不覆盖已有产物，不使用待定占位符。
