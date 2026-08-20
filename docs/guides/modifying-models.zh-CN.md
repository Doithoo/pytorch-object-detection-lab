# 修改检测模型

[English](modifying-models.md) | [外部工厂参考](adding-models.zh-CN.md)

本页通过一个较小的仓库示例，说明在保持数据、训练、checkpoint 和评估协议不变时，
detector 的哪些部分可以修改。它是参考实现，不代表性能结论。

## 示例文件

- `examples/extensions/custom_detector.py`：可导入的模型工厂。
- `configs/custom_detector_example.yaml`：对应的完整配置。

查看配置并离线执行一次参数更新：

```bash
uv run detect show-config --config configs/custom_detector_example.yaml
uv run detect train --config configs/custom_detector_example.yaml --dry-run --device cpu
```

示例使用 torchvision 的可复用组件构造 Faster R-CNN：

```text
MobileNet V3 Small features
-> 明确 out_channels 协议
-> 自定义 AnchorGenerator
-> MultiScaleRoIAlign
-> FasterRCNN
```

## 发生变化的部分

`width_mult` 改变 backbone 容量；anchor 尺寸和长宽比改变 RPN 直接表示的目标形状；
`min_size` 与 `max_size` 改变模型内部缩放。ROI pooling 和 Faster R-CNN heads
仍使用上游实现。

工厂从准备后的数据元数据接收 `num_classes`，因此不需要写死标签数量，可用于内置 VOC、
自定义 VOC 形状数据和 COCO JSON。

## 保持不变的部分

训练模式仍接收 `list[Tensor[3,H,W]]` 和对齐目标，并返回标量 loss 映射；评估模式仍
返回 `boxes`、`labels` 和 `scores`。这一边界让现有 trainer、安全 checkpoint 加载、
指标、错误分析和预测器无需增加模型专用分支。

## 可参考的修改点

- 将 `mobilenet_v3_small(...).features` 替换为其他特征提取器，并准确设置输出通道数。
- 查看目标尺寸统计后调整 anchor sizes。
- 数据中目标长期偏窄或偏宽时调整 aspect ratios。
- 表示方式确有变化时替换 ROI head。
- 一个工厂成为项目长期维护选项后再注册稳定名称；实验性工厂保持显式配置。

Dry run 只证明一次前向、反向和参数更新可以执行。客观结论还需要固定 manifest、可比设置、
验证指标、运行环境信息和保存的失败样例。
