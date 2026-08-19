# 工具脚本

[English](README.md) | [工作流教程](../docs/tutorial/README.zh-CN.md)

请在仓库根目录通过 `uv run python` 运行脚本。每个脚本只完成一项明确工作，不会静默训练或评估检测器。

## 脚本索引

| 文件 | 作用与前提 | 网络行为 | 预期产物或输出 |
|---|---|---|---|
| `download_data.py` | 下载并安全解压两份官方 VOC 2007 归档；需要数据目录写权限 | 本地没有校验通过的缓存归档时访问官方 VOC HTTP 地址，并校验发布的 MD5 | `<data-dir>/archives` 中的归档和解压后的 `VOCdevkit/VOC2007`；stdout 输出归档路径 |
| `preview_dataset.py` | 渲染准备后的 manifest 样本与框；需要 manifests 和本地源图像 | 离线 | PNG，默认路径为 `artifacts/dataset_preview.png`；stdout 输出路径 |
| `plot_metrics.py` | 绘制训练 `metrics.csv` 中所有 `loss*` 列；需要安装 `dev` extra 以提供 matplotlib | 离线 | 调用者指定的 PNG；stdout 输出路径，并拒绝空 CSV 或没有 loss 列的文件 |
| `generate_doc_assets.py` | 通过项目渲染代码重新生成确定性的合成教学图 | 离线；使用生成张量与默认内置字体 | 指定目录中的 `detection-target-anatomy.png` 和 `detection-error-analysis.png` |
| `__init__.py` | 让 `scripts` 成为可导入 package，供测试和辅助函数复用 | 不访问网络 | 没有命令，也不生成产物 |

## 工作流命令

下载脚本是正常路径中唯一可能访问网络的脚本：

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

它在解压前校验归档，并拒绝不安全成员或与本地冲突的文件。下载成功只代表完成源数据阶段，还需要运行 `detect prepare-data`，再检查数据准备结果：

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split train --limit 4 --output artifacts/dataset_preview.png
```

正常训练完成后，可以绘制 loss 列：

```bash
uv run python scripts/plot_metrics.py --metrics artifacts/run/metrics.csv --output artifacts/run/losses.png
```

文档图片只应通过显式维护命令重新生成：

```bash
uv run python scripts/generate_doc_assets.py --output-dir docs/assets
```

## 适用范围

下载与预览脚本用于检查源文件和标注，不能衡量模型质量。指标图只展示已经记录的 loss 列，不重新计算指标，也不能说明模型已经收敛。文档图片是合成教学图。这些输出都不是完整 VOC 成绩；已完成结果见 [Kaggle 训练记录](../docs/recorded-run/README.zh-CN.md)。
