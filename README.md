# E 题数据审计

这个项目只读审计 2026 年研究生数学建模竞赛 E 题的数据包。它不会训练模型、修改原始数据或解压整套数据；输出仅包含可追溯的 JSON、CSV 和 Markdown 契约。

## 依赖

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

## 运行

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

## 问题 1 特征提取

问题 1 以审计产物 `raw_samples.csv` 中的附件 1 映射为唯一清单，将每条原始视频的文本、语音和视觉证据对齐到 50 个原始物理时间槽。它只读取分卷归档中的一个视频进行处理，不会解压整包、修改原始归档或使用任何额外情感数据。此流程尚未在当前工作区执行真实提取：FFmpeg、7-Zip、模型和本地 NLTK 资产必须先由运行者准备。

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

WhisperX 的英文分词还要求 `model_cache/nltk_data` 中存在显式的 `punkt_tab`。以下命令是运行者在提取前一次性执行的准备动作，不是管线的运行时行为：

```bash
python3 -m nltk.downloader -d .tools/models/nltk_data punkt_tab
```

对一个尚不存在的输出目录，配置、FFmpeg、7-Zip、模型或 `punkt_tab` 的任一预检失败都会直接报错，且不会留下 Q1 输出目录。输出目录本身必须是新目录，不能覆盖此前的 smoke 或全量结果。

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

## 测试

```bash
PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest -v
```
