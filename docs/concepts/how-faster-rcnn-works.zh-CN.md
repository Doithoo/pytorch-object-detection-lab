# 本项目中的 Faster R-CNN 工作原理

[English](how-faster-rcnn-works.md) | [教程章节](../tutorial/03-faster-rcnn.zh-CN.md)

本页适合已经理解张量、但希望明确 Faster R-CNN 各部分职责的读者。内容解释公开的 torchvision 契约，不重新实现模型。

## 模型自有图像变换

调用者提供 `[0,1]` 范围的 RGB `list[float Tensor[3,Hi,Wi]]`。Torchvision 的通用检测器变换会归一化每张图像，根据注册的 `min_size`/`max_size` 策略缩放，把结果填充成内部批次，并记录每张图像尺寸。最终检测框会映射回调用者的原始坐标。数据集不应提前再应用另一套检测器缩放或归一化策略。

## 骨干网络与 FPN

骨干网络把像素转换为空间特征张量。MobileNet V3 Large 或 ResNet-50 决定特征提取器。特征金字塔网络把深层语义信息与较细空间细节融合，并输出多个具有统一通道接口的分辨率。

```text
填充图像 [B,3,H,W]
  -> 骨干网络各阶段
  -> FPN 侧向与自顶向下融合
  -> 特征图 {层级: [B,C,Hk,Wk]}
```

这些特征图是候选区域和 ROI 分类共享的证据，还不是检测框或 VOC 类别预测。

## 区域候选网络

RPN 在各 FPN 层级评估锚框，预测与类别无关的目标性和检测框调整，解码并裁剪候选框，排序并抑制重复，然后把每张图像数量可变的区域交给第二阶段。

它产生两项训练损失：

| 键 | 职责 |
|---|---|
| `loss_objectness` | 把采样锚框分类为目标或背景 |
| `loss_rpn_box_reg` | 把正锚框回归到匹配目标框 |

目标性不会选择 `person`、`dog` 或其他 VOC 类别。把 RPN 候选称为最终类别预测，会把职责错误地分配给第一阶段。

## ROI Align 与 ROI 头

ROI Align 从合适的金字塔层级为每个候选区域采样固定大小的特征。检测框头把这些特征转换成前景/背景类别逻辑值和类别专用检测框调整。后处理执行分数过滤、裁剪、非极大值抑制和每图数量限制。

它产生另外两项训练损失：

| 键 | 职责 |
|---|---|
| `loss_classifier` | 把 ROI 样本分类为背景或目标类别 |
| `loss_box_reg` | 把正 ROI 样本调整到最终目标框 |

因此 Faster R-CNN 的准确损失集合是 `loss_classifier`、`loss_box_reg`、`loss_objectness` 和 `loss_rpn_box_reg`。项目检查每项是否为有限标量，把全部返回损失求和为 `loss_total`，再反向传播。数值取决于数据、初始化、模型和设备，没有固定预期值。

## 随模式变化的公开接口

| 模式 | 调用 | 返回 |
|---|---|---|
| 训练 | `model.train(); model(images, targets)` | 四项标量损失的映射 |
| 评估 | 在 `torch.inference_mode()` 下执行 `model.eval(); model(images)` | 每张图像一个预测映射 |

每个预测包含 `boxes: float32 [M,4]`、`labels: int64 [M]` 和 `scores: float32 [M]`，每张图像的 `M` 可以不同。评估不能通过传入目标来请求损失；训练输出也不能当成指标预测解释。

离线运行真实契约：

```bash
uv run python examples/03_model_contract.py
```

预期输出会列出四项损失和三个预测键。示例用 `weights="none"` 和合成张量构造 `fasterrcnn_mobilenet_v3_large_320_fpn`，只执行前向传播，不更新参数，也不提供精度证据。

## 失败与证据边界

0 是背景，因此目标标签必须从 1 开始。检测框必须有限、有正面积，并采用零基连续 `xyxy`。图像与目标必须保持一一对应的列表。试运行在示例基础上对准备数据执行一次优化器更新；有界训练再加入验证和产物。两者都不是[单独记录的完整 VOC 实测结果](../recorded-run/README.zh-CN.md)。

接下来阅读[教程 04](../tutorial/04-training.zh-CN.md)了解优化和产物职责，或查看[模型参考](../reference/model-zoo.zh-CN.md)，比较两个 Faster R-CNN 骨干网络与 SSDLite。
