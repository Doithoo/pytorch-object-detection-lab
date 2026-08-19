# 比较两次训练

[English](experiments.md) | [Kaggle 训练记录](../recorded-run/README.zh-CN.md)

第一次 Kaggle 训练完成后，可以通过只改变一个设置来理解它的影响。项目已经提供一组
Faster R-CNN MobileNet 结果，可以把它作为已知起点，而不是排行榜。

## 从一个清楚的问题开始

合适的问题例如：

- 相同训练设置下，ResNet-50 backbone 与 MobileNet 有什么差异？
- 保持模型不变，学习率变化会怎样影响验证指标？
- Faster R-CNN 与 SSDLite 在相同数据和轮次下表现怎样？

一次不要同时改变模型、权重、学习率和轮次，否则结果很难解释。

## 保持这些内容不变

- train / valid / test 划分。
- 随机种子。
- 训练轮次和样本上限。
- 优化器、调度器和数据增强。
- 验证与测试指标。

每次使用不同的 `run_name` 和输出目录，保留各自的 `config.yaml` 和 `metrics.csv`。

## 先检查配置

例如只更换模型：

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set run_name experiment-a
uv run detect show-config --config configs/learning_minimal.yaml --set run_name experiment-b --set model.name ssdlite320_mobilenet_v3_large
```

本地可以先做 dry run，确认两种模型都能读取数据并更新一次：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name experiment-a --dry-run --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name experiment-b --set model.name ssdlite320_mobilenet_v3_large --dry-run --device cpu
```

dry run 不保存模型，也不比较精度。正式对比建议在 Kaggle GPU 上使用相同的数据和训练
预算运行两份配置。

## 比较验证指标

两次训练完成后：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

命令会为每个运行选择最佳验证行，并列出重要配置差异。除了排名，还要一起看：

- `metrics.csv` 中的变化趋势。
- 每类 AP 和召回率。
- 误检与漏检图。
- 训练时间和显存是否适合你的使用场景。

## 最后再看测试集

使用 valid 选择配置和 checkpoint。确定所有选择后，再对胜出的设置评估一次 test。不要
根据 test 结果继续反复修改设置。

带样本上限或很少轮次的运行只能说明那次小规模尝试发生了什么。项目当前发布的完整 VOC
结果仍只有 [Kaggle v7 训练](../recorded-run/README.zh-CN.md)。
