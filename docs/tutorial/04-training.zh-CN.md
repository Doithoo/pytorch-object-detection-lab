# 教程 04：在 Kaggle 训练 Faster R-CNN

[English](04-training.md) | [教程索引](README.zh-CN.md)

这一章把前面的数据和模型放到一次完整训练中。推荐直接使用 Kaggle T4；完整 VOC 训练不
适合用普通 CPU 等待。

## 这次训练使用什么

参考配置是 [`../../configs/reference_fasterrcnn.yaml`](../../configs/reference_fasterrcnn.yaml)：

- Faster R-CNN MobileNet V3 Large 320 FPN。
- ImageNet1K V1 backbone 权重。
- 官方 VOC 2007 train / valid / test 划分。
- 26 个 epoch，SGD 优化器和 step 学习率调度。
- 验证集 `map_50_95` 选择 `best.pt`。

Kaggle runner 会把设备改为 CUDA、开启 AMP、使用两个 data workers，并把路径放到
`/kaggle/working`。模型结构和优化设置保持不变。

## 提交训练

先完成 [Kaggle 指南](../guides/kaggle.zh-CN.md)中的登录和账户名修改，然后运行：

```bash
kaggle kernels push -p docs/recorded-run/kaggle
```

网页中确认 T4 或更新 GPU、Internet 开启。runner 会自动完成数据下载、准备和一个 batch
的 dry run，不需要在 Kaggle 单独执行本地命令。

## dry run 在检查什么

正式 epoch 前，runner 会读取一批真实 VOC 图像并完成一次参数更新。诊断信息包含：

```text
image_shapes=((3, H1, W1), (3, H2, W2))
target_counts=(N1, N2)
loss_total=<有限值>
loss_classifier=<有限值>
loss_box_reg=<有限值>
loss_objectness=<有限值>
loss_rpn_box_reg=<有限值>
dry-run OK
```

`dry-run OK` 表示数据可以进入模型、loss 可以反向传播、优化器可以更新参数。它不是训练
结果；真正的训练从后面的 epoch 日志开始。

## 看懂 epoch 日志

每轮包含两个阶段：

1. train：计算 loss、反向传播并更新模型。
2. valid：不更新参数，计算验证集 mAP 和召回率。

训练或评估时间较长时，runner 每 60 秒输出心跳。只要心跳继续出现，就不要停止或重新
提交。完整 v7 运行的训练时间是 3,025.660 秒，约 50 分钟。

每轮结果也会写入 `metrics.csv`。阅读时先关注：

| 列 | 含义 |
|---|---|
| `epoch` | 已完成轮次 |
| `loss_total` | 四项训练 loss 的和 |
| `valid_map_50_95` | 选择最佳 checkpoint 的主要验证指标 |
| `valid_map_50` | IoU 0.5 下的验证 AP |
| `learning_rate` | 当前学习率 |

loss 是优化目标，mAP 是验证集检测表现，它们不必同步升降。

## 为什么保存 best.pt 和 last.pt

- `best.pt`：截至当前验证 `map_50_95` 最高的轮次，适合最终评估和预测。
- `last.pt`：最近完成的轮次，包含继续训练所需的优化器、调度器和随机状态。

已发布运行在第 18 轮取得最佳验证 `map_50_95 = 0.313245`，但仍按计划完成第 26 轮。
最终测试使用第 18 轮的 `best.pt`，不是第 26 轮的 `last.pt`。

## 训练结束后会得到什么

`artifacts/reference-fasterrcnn/` 中包含：

```text
config.yaml
run.yaml
metrics.csv
best.pt
last.pt
evaluation/
```

`config.yaml` 是实际运行配置，`run.yaml` 记录设备、数据和版本，`metrics.csv` 是完整训练
曲线。不要只下载 checkpoint；这些小文件能帮助你理解模型是怎样得到的。

## 可选：在本地完成一次小检查

有本地 VOC 数据但没有 GPU 时，可以只检查一批数据和一次更新：

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

`learning_minimal.yaml` 使用随机权重和少量样本。它适合确认代码路径，不作为训练结果。

## 可选：使用本地 GPU 完整训练

已有兼容 CUDA GPU、完整 VOC 数据和 backbone 权重下载能力时，可以运行：

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml --device cuda
```

新的运行需要使用尚不存在的输出目录。要继续中断的运行，使用 `last.pt`：

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml --resume artifacts/reference-fasterrcnn/last.pt --device cuda
```

续训时模型、数据和影响训练含义的设置必须与 checkpoint 一致。完整字段说明见
[checkpoint 参考](../reference/checkpoint-schema.zh-CN.md)。

下一步进入[评估与预测](05-evaluation-and-inference.zh-CN.md)，使用真实 Kaggle 文件分析
模型做对和做错的地方。
