# E 题数据审计与问题 1 特征对齐

这是一个面向 2026 年研究生数学建模竞赛 E 题的可复现数据入口。它把原始分卷 ZIP 作为只读输入：先审计附件 1 至附件 4 的数据契约，再为问题 1 从附件 1 的 100 条原始视频生成可追溯的三模态时间序列特征。

## 项目提供什么

| 命令 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| `audit` | 一套分卷 ZIP 与显式 7-Zip 路径 | JSON、CSV、Markdown 审计契约 | 核验成员、视频-标注映射、特征字段和专项样本结构 |
| `extract-q1` | 已审计映射、同一 ZIP、显式本地工具与模型 | 压缩 NPZ 特征、时序证据、覆盖表与运行清单 | 将文本、语音、视觉统一到 50 个原始物理时间槽 |

它不会训练情感模型、修改原始数据、解压整套数据，也不会在运行时下载模型或替代数据。附件 2 及专项集的审计结果与问题 1 的原始视频特征提取保持分开，避免混淆数据用途。

## 处理边界

- 原始归档只通过显式 `7za`、`7z` 或 `7zz` 读取；问题 1 一次仅处理一个视频。
- 成功样本保存固定形状的 `text[50, 768]`、`audio[50, 50]` 和 `vision[50, 56]`；时间边界和各模态掩码单独保存，零值不代表模态缺失。
- 单样本失败会留在覆盖表并令命令返回非零状态，不会伪造特征或中断其余样本。
- 所有运行路径均要求显式配置。本地模型、`punkt_tab`、FFmpeg 或 7-Zip 缺失时，预检直接失败且不创建半成品 Q1 输出。

## 仓库结构

```text
src/e_mosei_audit/        审计、归档读取与问题 1 提取实现
tests/                    单元测试、假依赖端到端测试与 opt-in 真实 smoke
docs/q1-config.example.toml
                           问题 1 的显式本地路径模板
artifacts/                忽略的派生审计和特征产物
.tools/                   忽略的本地 7-Zip、FFmpeg、Python 依赖与模型缓存
```

## 数据审计环境

常规环境使用 Python 3.12 及以上版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

如果系统 Python 没有 `ensurepip`，可将 Excel 依赖安装到忽略的本地工具目录，并从源码启动：

```bash
python3 -m pip install --target .tools/python 'numpy>=1.26' 'openpyxl>=3.1' 'pandas>=2.2'
export PYTHONPATH="$PWD/src:$PWD/.tools/python"
```

分卷 ZIP 需要显式提供 `7za`、`7z` 或 `7zz`。在没有管理员权限的 Debian/Ubuntu 环境中，可准备一个本地 `7za`：

```bash
mkdir -p .tools/bin .tools/pkg
cd .tools/pkg
apt-get download 7zip
dpkg-deb --fsys-tarfile ./7zip_*.deb | tar -xOf - ./usr/lib/7zip/7za > ../bin/7za
chmod +x ../bin/7za
cd ../..
```

`.tools/` 不进入 Git，也不应作为竞赛提交物。

## 执行数据审计

安装为包后：

```bash
e-mosei-audit audit \
  --archive 'E题数据 (2).zip' \
  --seven-zip .tools/bin/7za \
  --output artifacts/data-audit
```

源码启动时：

```bash
PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m e_mosei_audit.cli audit \
  --archive 'E题数据 (2).zip' \
  --seven-zip .tools/bin/7za \
  --output artifacts/data-audit
```

输出目录必须不存在。完成后将得到：

- `manifest.json`：归档成员及其大小。
- `raw_samples.csv`：附件 1 的标签、视频和时长映射。
- `feature_contract.json`：附件 2 各划分的字段、形状、标签、非有限值和零段摘要。
- `special_samples.csv`：附件 3/4 的样本字段与零段证据。
- `audit_report.md`：异常、警告和建模解释边界。

连续全零位置仅作为数值证据记录，不会在没有独立长度或掩码字段时被断言为赛题注入的模态缺失。

## 问题 1：多模态特征提取

问题 1 以审计产物 `raw_samples.csv` 中的附件 1 映射为唯一清单，将每条原始视频的文本、语音和视觉证据对齐到 50 个原始物理时间槽。它只读取分卷归档中的一个视频进行处理，不会解压整包、修改原始归档或使用任何额外情感数据。

本工作区已在显式本地工具和离线模型缓存下完成一次 100 条真实运行：覆盖表、NPZ 和对齐证据均为 100 条，全部成功。该运行的派生产物保留在被 Git 忽略的 `artifacts/` 中；下述准备步骤仍是其他机器复现所必需的条件。

### 本地准备

问题 1 的可选依赖安装到忽略的本地目录，源码运行时使用该目录：

```bash
python3 -m pip install --target .tools/python '.[q1]'
export PYTHONPATH="$PWD/src:$PWD/.tools/python"
```

复制 [q1-config.example.toml](docs/q1-config.example.toml) 到仓库根目录的 `q1.toml`，并保留或改为实际的显式本地路径：

```toml
[paths]
audit_dir = "artifacts/data-audit-v2"
archive = "E题数据 (2).zip"
seven_zip = ".tools/bin/7za"
ffmpeg = ".tools/bin/ffmpeg"
model_cache = ".tools/models"
output_dir = "artifacts/q1-default"
```

其中 `seven_zip` 和 `ffmpeg` 必须是存在且可执行的本地文件，`audit_dir` 必须含有已审计的 `raw_samples.csv`，`archive` 必须是该审计对应的分卷 ZIP 最后 `.zip` 卷。`model_cache` 必须预先包含本地 `facebook/wav2vec2-base-960h` 对齐权重、`bert-base-uncased`、`face_landmarker.task`；流程只从这些路径加载，不会自动下载模型或在资产缺失时生成替代特征。

WhisperX 的英文分词还要求 `model_cache/nltk_data` 中存在显式的 `punkt_tab`。下载器的 `-d` 目标必须是 **`q1.toml` 中实际 `model_cache` 值的 `nltk_data` 子目录**，即 `<the exact q1.toml model_cache value>/nltk_data`；不能在修改 `model_cache` 后仍沿用默认目录。以下是上面默认 `model_cache = ".tools/models"` 时、运行者在提取前一次性执行的可复制准备命令，不是管线的运行时行为：

```bash
python3 -m nltk.downloader -d .tools/models/nltk_data punkt_tab
```

对一个尚不存在的输出目录，配置、FFmpeg、7-Zip、模型或 `punkt_tab` 的任一预检失败都会直接报错，且不会留下 Q1 输出目录。输出目录本身必须是新目录，不能覆盖此前的 smoke 或全量结果。

### 原始转写与对齐回退

BERT 始终编码 Excel 中的原始转写，审计 CSV 和证据中的 `transcript` 也始终保留原文。WhisperX 先使用原文做词级对齐；只有它明确返回未定时词时，才会进行一次确定性的对齐专用回退：弯引号、破折号和纯标点不作为发音词传入，阿拉伯数字会展开为英文基数词或序数词。回退后的词序必须与该规范文本完全一致，否则样本仍记为失败，不会通过时间插值伪造对齐。

发生回退时，`alignment_evidence/<sample_id>.json` 的 `original_words` 保留原始字符范围和原文片段，并额外写入 `aligned_text`，例如 `2008, -> two thousand eight`。该字段不存在即表示该原始词未经过规范化；它不是新的训练文本，也不改变原始标注。

### 先 smoke，再全量

确认上述 FFmpeg、7-Zip、模型和 NLTK 资产均已就绪后，先只运行一个已审计样本：

```bash
python3 -m e_mosei_audit.cli extract-q1 \
  --config q1.toml \
  --output artifacts/q1-smoke \
  --limit 1
```

先检查 `q1_samples.csv`、对应的 `features/*.npz`、`alignment_evidence/*.json` 与 `typical_sample.md`，并检查 smoke 产物大小：

```bash
du -sh artifacts/q1-smoke
```

确认时间槽证据和大小后才允许运行 100 条全量提取：

```bash
python3 -m e_mosei_audit.cli extract-q1 \
  --config q1.toml \
  --output artifacts/q1-full
```

全量运行必须使 `q1_samples.csv` 恰有 100 条记录。每条成功记录关联 `features/<sample_id>.npz`（`text[50, 768]`、`audio[50, 50]`、`vision[50, 56]`、掩码和 50 个原始秒数边界）及 `alignment_evidence/<sample_id>.json`（转写词/token、音频窗口、视觉帧与槽位的对应）。`feature_contract.json` 固定维度、dtype、模型和参数，`run_manifest.json` 与 `run.log` 记录输入指纹、工具版本和执行计数，`q1_summary.json` 汇总覆盖/成功/失败数，`typical_sample.md` 展示一个成功样本的可回查证据。

单样本失败不会中断其余样本：它仍会在 `q1_samples.csv` 中留下带失败原因的覆盖行，命令以非零状态结束。零值与模态不可用状态分别由特征值和独立掩码表达，不能互相替代。

`.tools/`（包括 Python 依赖、模型和 NLTK 缓存）、`artifacts/`、`models/` 以及解码期间的临时 WAV/PNG 都是忽略项，不能提交；中间媒体在单样本完成后删除。最终代码、说明与选择保留的交付产物总大小目标不超过 50 MB，先通过 smoke 的 `du -sh` 检查再决定是否生成全量特征。

## 验证

```bash
PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest -v
```

默认测试覆盖纯数值契约、归档边界、媒体解码命令、输出事务、适配器的本地资产约束与假依赖端到端流程。真实归档审计和真实问题 1 smoke 均为 opt-in 测试：只有显式提供 `E_MOSEI_ARCHIVE`、`E_MOSEI_7ZA` 或 `E_MOSEI_Q1_CONFIG` 后才会执行；未配置时跳过是预期行为。当前工作区已显式执行真实问题 1 smoke 与 100 条全量运行，实际结果见上文，但这些派生产物不会提交到仓库。
