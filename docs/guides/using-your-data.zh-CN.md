# 使用自己的 VOC 形状数据

[English](using-your-data.md) | [教程：数据与检测框](../tutorial/02-data-and-boxes.zh-CN.md)

本指南适合能够整理成 Pascal VOC 2007 目录和标注结构的数据。当前程序不能只靠配置接受任意类别或任意标注格式；这两种扩展都需要修改代码。项目目前没有稳定的外部数据集插件接口。

## 选择正确路径

需要遵循 VOC 2007 官方协议和样本数时，下载两个经过校验的压缩包，并且不要使用例外参数：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
```

只有当受控数据有意采用 VOC 的目录、XML、类别和坐标规则，但划分数量不同时，才使用夹具路径：

```bash
uv run detect prepare-data --data-dir data/my-voc --manifest-dir data/my-manifests --allow-nonstandard-counts
```

`--allow-nonstandard-counts` 只关闭 train 2501 张、valid 2510 张和 test 4952 张的数量检查，其他校验仍会执行。这样生成的清单只能作为 VOC 形状数据的证据，不能称为官方 VOC 2007 结果。

## 源目录要求

```text
data/my-voc/VOCdevkit/VOC2007/
  JPEGImages/<image-id>.jpg
  Annotations/<image-id>.xml
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
  ImageSets/Main/test.txt
```

划分文件的每个非空行都是不带扩展名的图像标识。同一划分内不能重复，train、valid（读取 `val.txt`）和 test 之间也不能重叠。每个标识必须同时对应可解码的 JPEG 和 XML。XML 中的 `filename` 必须等于 `<image-id>.jpg`，宽高必须与解码图像一致。

每个 XML 目标名称必须属于[数据集格式参考](../reference/dataset-format.zh-CN.md)列出的 20 类。要使用任意类别，必须修改 `VOC_CLASSES`、元数据构造、测试和类别数契约。`difficult` 可以省略，默认是 `0`；若存在，只能是 `0` 或 `1`。

VOC XML 检测框采用一基且端点包含的坐标。准备阶段只转换一次：`(xmin, ymin, xmax, ymax)` 变为 `(xmin - 1, ymin - 1, xmax, ymax)`，随后裁剪到图像边界，并拒绝非正宽高的框。运行时目标采用零基连续 `xyxy`，最大边界不包含对应像素。

## 发布并检查清单

准备成功后会打印 `identity=` SHA-256 和划分数量，并原子发布 `train.csv`、`valid.csv`、`test.csv`、`dataset.yaml`、`source.yaml` 与 `summary.txt`。这些文件保存路径、元数据、哈希和来源，不会复制 JPEG 或 XML 内容；运行时仍从 `data.data_dir` 读取源文件。

```bash
uv run detect inspect-data --manifest-dir data/my-manifests --data-dir data/my-voc --split train --limit 16
uv run python scripts/preview_dataset.py data/my-manifests --data-dir data/my-voc --split valid --limit 4 --output artifacts/my-data-preview.png
```

预期证据包括一份 YAML 检查报告和 `artifacts/my-data-preview.png`。若 `--output` 指向的 PNG 已存在，预览脚本会直接覆盖且不询问；需要保留旧证据时应换一个输出路径。训练前应核对类别名称、空图像、困难目标数量、图像范围和检测框位置。训练会移除困难目标；检查和评估则保留它们，并设置 `difficult=True`、`iscrowd=1`。

## 证明配置后的运行路径

让学习配方指向同一源目录与清单目录。先检查解析值，以及它们对应的 `cli` 来源：

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-check
```

随后用一个试运行批次跨越数据集、模型、损失和优化器边界：

```bash
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-dry-run --dry-run --device cpu
```

预期输出会列出图像形状、目标数量、有限损失和 `dry-run OK`。图像与 XML 从 `data/my-voc` 下读取，清单行与标识来自 `data/my-manifests`。试运行不会写正常运行目录。

若要检查有界产物路径，请使用不同的名称和相同覆盖：

```bash
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-bounded --device cpu
```

成功后，标准运行产物会写入 `artifacts/my-data-bounded`。学习配方仍限制样本和轮次，非标准数量数据也仍然不是官方 VOC 2007。这些命令只提供集成证据，不能产生完整基准结果。

只要源图像、XML 或划分发生变化，就重新准备数据并使用新的标识。不要手工修改 CSV 或哈希来保留旧标识。接下来可阅读[试运行与训练教程](../tutorial/04-training.zh-CN.md)；若源格式确实不同，则阅读[添加数据集](adding-datasets.zh-CN.md)。
