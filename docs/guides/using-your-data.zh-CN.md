# 使用自己的 VOC 格式数据

[English](using-your-data.md) | [数据格式](../reference/dataset-format.zh-CN.md)

如果数据已经使用 Pascal VOC 的 JPEG、XML 和划分文本格式，可以复用项目的数据准备、
训练和评估代码。本指南适合仍使用项目内置 20 个 VOC 类别的数据。

要支持任意新类别，需要修改 `VOC_CLASSES`、类别数、模型 checkpoint 信息和测试，这不在
本页范围内。

## 目录结构

```text
my-data/
└── VOCdevkit/
    └── VOC2007/
        ├── Annotations/
        ├── ImageSets/Main/
        │   ├── train.txt
        │   ├── val.txt
        │   └── test.txt
        └── JPEGImages/
```

每个图像 ID 需要同名 `.jpg` 和 `.xml`。XML 中的类别必须属于 VOC 20 类，`difficult`
可以省略或为 `0` / `1`。坐标使用 VOC 的 1-based inclusive 格式。

## 准备并查看数据

非官方数量需要显式允许：

```bash
uv run detect prepare-data --data-dir my-data --manifest-dir data/my-manifests --allow-nonstandard-counts
uv run detect inspect-data --manifest-dir data/my-manifests --data-dir my-data --split train --limit 16
uv run python scripts/preview_dataset.py data/my-manifests --data-dir my-data --split train --limit 4 --output artifacts/my-data-preview.png
```

打开预览图，确认类别和框位置。数量、类别、坐标或图片尺寸错误时，应先修复源 XML，再重新
生成清单。

## 用少量样本检查训练

把配置路径覆盖为自己的目录：

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set data.data_dir my-data --set data.manifest_dir data/my-manifests --set run_name my-data-check
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir my-data --set data.manifest_dir data/my-manifests --set run_name my-data-check --dry-run --device cpu
```

看到 `dry-run OK` 后，说明一批图像和标注可以完成一次更新。它不保存模型。

## 开始一次训练

先根据数据规模复制并调整配置，设置新的 `run_name`，再在 Kaggle 或本地兼容 GPU 上运行。
非官方数据产生的指标只能描述你自己的划分，不能与项目的 VOC 2007 数字直接比较。

训练后保存 `config.yaml`、`run.yaml`、`metrics.csv`、`best.pt` 和 `last.pt`。评估时继续
使用生成该 checkpoint 的同一份 `data/my-manifests`。
