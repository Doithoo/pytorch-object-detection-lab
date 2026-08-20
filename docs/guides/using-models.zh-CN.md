# 选择模型

[English](using-models.md) | [模型参考](../reference/model-zoo.zh-CN.md)

项目提供五个 torchvision 检测器和显式外部工厂。Faster R-CNN MobileNet 有一份 Kaggle 实测记录，其他注册项在相同项目协议下展示不同架构。

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

这些命令只显示模型信息，不下载权重，也不开始训练。

## 模型特点

| 模型 | 特点 | 建议 |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 两阶段、较紧凑，已完成 Kaggle VOC 训练 | 已记录的参考实现 |
| `fasterrcnn_resnet50_fpn` | 两阶段、更大的 backbone，需要更多计算 | 在 Faster R-CNN 内观察 backbone 容量差异 |
| `retinanet_resnet50_fpn` | anchor-based 单阶段模型，使用 Focal Loss | 查看密集 anchors 和类别不平衡处理 |
| `fcos_resnet50_fpn` | anchor-free 单阶段模型，使用 centerness | 查看不依赖预定义 anchors 的位置预测 |
| `ssdlite320_mobilenet_v3_large` | 紧凑单阶段模型，固定 320 输入 | 查看面向移动端的密集检测器 |

项目没有在相同训练预算下发布五个模型的完整对比，因此不能从这张表推断速度或精度排名。

## 权重设置

- `weights: imagenet1k_v1`：检测 head 从随机状态开始，backbone 使用 ImageNet 预训练权重。
  Kaggle 参考训练使用这一设置，需要联网下载一次。
- `weights: none`：检测器和 backbone 都随机初始化，适合离线示例和 dry run。

`none` 不等于“不需要数据”，只表示构造模型时不下载预训练权重。

## 先查看最终参数

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

模型参数中的 `min_size`、`max_size` 和 `box_score_thresh` 会影响内部缩放和预测过滤。对比某项设置时，一次只改变与该问题相关的参数。

## 做一次公平比较

比较模型时保持数据划分、权重策略、随机种子、优化器、轮次和样本限制一致，只改变
`model.name`。先用 dry run 确认模型能执行，再在 Kaggle 上给两个训练相同的 GPU 时间。

```bash
uv run detect train --config configs/learning_minimal.yaml --set model.name ssdlite320_mobilenet_v3_large --dry-run --device cpu
```

dry run 只检查一次更新。完整比较还需要验证指标、逐类结果、运行时间和错误图。步骤见
[比较训练](experiments.zh-CN.md)。
