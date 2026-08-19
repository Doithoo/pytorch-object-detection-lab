# 选择模型

[English](using-models.md) | [模型参考](../reference/model-zoo.zh-CN.md)

项目提供三个 torchvision 检测器。第一次训练建议使用已经在 Kaggle 跑通的 Faster R-CNN
MobileNet，熟悉流程后再比较其他模型。

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

这些命令只显示模型信息，不下载权重，也不开始训练。

## 三个模型怎样选择

| 模型 | 特点 | 建议 |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 两阶段、较紧凑，已完成 Kaggle VOC 训练 | 第一次运行使用 |
| `fasterrcnn_resnet50_fpn` | 两阶段、更大的 backbone，需要更多计算 | 理解 backbone 影响时比较 |
| `ssdlite320_mobilenet_v3_large` | 单阶段、固定 320 输入 | 比较一阶段和两阶段检测器时使用 |

项目没有在相同训练预算下发布三个模型的完整对比，因此不能从这张表推断速度或精度排名。

## 权重设置

- `weights: imagenet1k_v1`：检测 head 从随机状态开始，backbone 使用 ImageNet 预训练权重。
  Kaggle 参考训练使用这一设置，需要联网下载一次。
- `weights: none`：检测器和 backbone 都随机初始化，适合离线示例和 dry run。

`none` 不等于“不需要数据”，只表示构造模型时不下载预训练权重。

## 先查看最终参数

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

模型参数中的 `min_size`、`max_size` 和 `box_score_thresh` 会影响内部缩放和预测过滤。第一次
运行先使用项目默认值，不要同时修改多个参数。

## 做一次公平比较

比较模型时保持数据划分、权重策略、随机种子、优化器、轮次和样本限制一致，只改变
`model.name`。先用 dry run 确认模型能执行，再在 Kaggle 上给两个训练相同的 GPU 时间。

```bash
uv run detect train --config configs/learning_minimal.yaml --set model.name ssdlite320_mobilenet_v3_large --dry-run --device cpu
```

dry run 只检查一次更新。完整比较还需要验证指标、逐类结果、运行时间和错误图。步骤见
[比较训练](experiments.zh-CN.md)。
