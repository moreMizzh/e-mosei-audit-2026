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
python3 -m pip install --target .tools/python 'openpyxl>=3.1'
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

## 测试

```bash
PYTHONPATH="$PWD/src:$PWD/.tools/python" python3 -m pytest -v
```
