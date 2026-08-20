# PyTorch 目标检测入门

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-object-detection-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**English: [README.md](README.md)**

这是一个面向初学者的 PyTorch 目标检测项目。你将使用 torchvision 的 Faster R-CNN
学习边界框、VOC 数据、训练、评估和预测，并在 Kaggle 的免费 GPU 上完成一次真实训练。

不需要先配置本地 CUDA，也不需要先读完所有参考文档。建议从 Kaggle 开始，遇到概念时
再回到对应教程。

## 已完成的 Kaggle 训练

项目已经在 Kaggle Tesla T4 上完成 26 轮 VOC 2007 训练。模型根据验证集表现选择第 18
轮的 `best.pt`，最后一次性评估 4,952 张测试图像。

| 项目 | 结果 |
|---|---:|
| 模型 | Faster R-CNN MobileNet V3 Large 320 FPN |
| 测试 `mAP@0.5:0.95` | **0.322312** |
| 测试 `mAP@0.5` | **0.609917** |
| 训练时间 | 3,025.660 秒，约 50 分钟 |
| Kaggle 任务总时间 | 3,223.9 秒，约 54 分钟 |

![Kaggle 训练所得模型在 VOC 2007 测试图像上的预测](docs/recorded-run/evaluation/visualizations/summary.png)

这是已完成的 Kaggle 训练中保存的真实预测，不是示意图。完整指标、各类别结果、误检和漏检
案例见 [Kaggle 训练记录](docs/recorded-run/README.zh-CN.md)。仓库只发布这一组完整训练
结果；小样本运行和合成示例只用于理解代码。

## 在 Kaggle 开始训练

你需要一个能够使用 GPU 的 Kaggle 账户。项目提供的 runner 会上传源码、下载官方 VOC
2007、准备数据、训练、评估并保存结果。整个过程遵循：

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

### 1. 获取项目并安装 Kaggle CLI

```bash
git clone https://github.com/Doithoo/pytorch-object-detection-lab.git
cd pytorch-object-detection-lab
uv tool install kaggle
kaggle auth login
```

Kaggle CLI 是提交和下载工具，不会加入项目的训练依赖。这个 runner 不需要
`kagglehub`，也不需要额外挂载 Kaggle Dataset。

### 2. 改成你的 Kaggle 用户名

打开 `docs/recorded-run/kaggle/kernel-metadata.json`，把 `id` 中的 `yashowhoo` 改成你的
Kaggle 用户名。保留 `enable_gpu: true` 和 `enable_internet: true`。

### 3. 提交并查看运行状态

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-object-detection-lab-voc2007-gpu
```

然后在 Kaggle 网页打开该任务。确认分配的是 T4 或更新的 GPU。页面可能显示 T4 x2，
但本项目是单 GPU 训练，只使用其中一张，这是正常现象。日志每 60 秒会输出一次心跳，
完整运行约需 50-60 分钟。

### 4. 下载结果

状态变成 `COMPLETE` 后，只下载训练产物，避免把临时 VOC 数据一起下载：

```bash
kaggle kernels output <你的用户名>/pytorch-object-detection-lab-voc2007-gpu --file-pattern 'artifacts/.*' -p kaggle-output
```

从 `kaggle-output` 开始查看：

- `metrics.csv`：每一轮的训练损失和验证指标。
- `best.pt`：验证集表现最好的模型。
- `last.pt`：最后一轮模型和续训状态。
- `evaluation/evaluation.json`：测试集汇总指标。
- `evaluation/per_class.csv`：20 个 VOC 类别的结果。
- `evaluation/visualizations/`：真实预测、误检和漏检图。

从账户准备到故障处理的完整步骤见 [Kaggle 训练指南](docs/guides/kaggle.zh-CN.md)。

## 推荐学习顺序

不必一次读完所有页面。按下面的顺序边运行边看即可：

1. [先看学习路线](docs/tutorial/learning-path.zh-CN.md)，了解每一步在做什么。
2. [理解图像、标签和边界框](docs/tutorial/00-basics.zh-CN.md)。
3. [认识 VOC 数据](docs/tutorial/02-data-and-boxes.zh-CN.md)。
4. [理解 Faster R-CNN](docs/tutorial/03-faster-rcnn.zh-CN.md)。
5. [在 Kaggle 训练](docs/tutorial/04-training.zh-CN.md)。
6. [读懂评估与预测](docs/tutorial/05-evaluation-and-inference.zh-CN.md)。

完整导航见[文档首页](docs/README.zh-CN.md)。

## 可选：先在本地检查代码

如果你想在提交 Kaggle 前确认环境和命令，可以在 Python 3.10-3.12 环境中运行：

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect list-models
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

这些命令不会开始完整训练。准备好本地 VOC 数据后，还可以用 CPU 完成一次小批次更新：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

本地完整训练只推荐给已有兼容 GPU 的读者，命令和注意事项放在
[训练教程的可选章节](docs/tutorial/04-training.zh-CN.md)中。

## 项目中可以学到什么

- VOC 的 XML 标注如何变成 torchvision 使用的 `boxes`、`labels` 和 `image_id`。
- 为什么目标检测 batch 是图像列表和标注列表，而不是一个固定尺寸张量。
- Faster R-CNN 训练时为什么返回多项 loss，评估时为什么返回框、类别和分数。
- 如何用验证集选择 `best.pt`，再用测试集报告最终结果。
- 如何查看逐类别 AP、误检、漏检和预测图，而不只看一个总分。
- 如何从 checkpoint 继续训练，或对自己的图片运行预测。

项目提供三个模型配置：

| 名称 | 适合用途 |
|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 默认入门模型，也是 Kaggle 实测模型 |
| `fasterrcnn_resnet50_fpn` | 更大的 Faster R-CNN 对照模型 |
| `ssdlite320_mobilenet_v3_large` | 单阶段检测器对照模型 |

模型和权重说明见[模型选择指南](docs/guides/using-models.zh-CN.md)，配置文件说明见
[配置目录](configs/README.zh-CN.md)。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
```

测试使用合成数据和临时文件，不会下载 VOC 或预训练权重。贡献代码前请阅读
[贡献指南](CONTRIBUTING.zh-CN.md)。项目采用 [MIT License](LICENSE)。

<!-- Documentation path: download -> prepare -> inspect -> dry run -> train -> evaluate -> predict | recorded full-VOC score 0.322312 -->
