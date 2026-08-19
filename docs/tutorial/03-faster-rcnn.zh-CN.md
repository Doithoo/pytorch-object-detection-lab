# 教程 03：把 Faster R-CNN 看成张量流水线

[English](03-faster-rcnn.md) | [教程索引](README.zh-CN.md)

前提是理解教程 00 和 02 中基于列表的批次与 `xyxy` 标注契约。本章只使用随机合成输入和
`weights=none`，不需要 VOC、检查点或网络。目标是分清各模块职责和随模式变化的
API，而不是重新实现 torchvision 内部细节。

## 检测器负责填充与缩放

调用者传入 `list[Tensor[3, Hi, Wi]]`。torchvision 的通用检测变换层会分别归一化、缩放
各张图像，再将它们填充成内部图像批次张量 `[B,3,Hpad,Wpad]`，同时保留每张缩放后图像
的尺寸。最终框需要这些元数据才能映射回调用者坐标。

标注列表与图像列表一一对应。训练时，每份标注提供 `[Ni,4]` 前景框和 `[Ni]` 标签；
评估时不传标注。

## 骨干网络与 FPN：从像素到多尺度特征

骨干网络将填充后的图像张量变成空间特征图。Feature Pyramid Network（FPN）融合深层
语义与较细空间分辨率，概念上输出多个尺度：

```text
image list [B, 3, Hpad, Wpad]
    -> 骨干网络 + FPN
第 k 层特征 [B, C, Hk, Wk]
```

小目标可以使用更细的层，大目标可以使用更粗的层。这些特征还不是框，也没有最终 VOC
类别判断。

## RPN：从特征图到类别无关的候选框

Region Proposal Network（RPN）在各 FPN 层评估锚框，为每个锚框预测目标性和框调整量。
经过解码、裁剪、排序与抑制后，下一阶段会为每张图接收数量不同的 `[Ki,4]` 候选框。

RPN 产生两个训练值：

- `loss_objectness`：采样锚框中是否包含目标，而不是背景。
- `loss_rpn_box_reg`：正锚框向真实目标调整得是否准确。

这些候选框在这一阶段没有类别。过早把它们称为 person 或 dog，会混淆 RPN 与 ROI heads
的职责。

## ROI heads：从候选框到类别与最终框

ROI Align 从适当的 FPN 层为每个候选框抽取固定空间尺寸的特征。box head 产生包含背景的
类别 logits 和类别相关的框调整量，后处理再为每张图返回数量不同的结果：

```text
prediction["boxes"]   float32 [M, 4]
prediction["labels"]  int64   [M]
prediction["scores"]  float32 [M]
```

ROI heads 产生另外两个训练值：

- `loss_classifier`：背景、前景以及具体目标类别的分类损失。
- `loss_box_reg`：对正 ROI 样本的最终框精修损失。

因此，Faster R-CNN 恰好返回以下四个 torchvision 损失键：`loss_classifier`、
`loss_box_reg`、`loss_objectness`、`loss_rpn_box_reg`。项目将它们求和为 `loss_total`，
用于反向传播和日志。数值取决于初始化、输入和设备，本教程不预设任何损失值。

## 训练模式与评估模式使用不同 API

torchvision 检测模型会同时改变输入要求和返回值：

| 模式 | 调用 | 返回 |
|---|---|---|
| 训练 | `model(images, targets)` | 标量损失张量字典 |
| 评估 | 推理模式下 `model(images)` | 每张图一个预测字典 |

运行真实维护模型的契约：

```bash
uv run python examples/03_model_contract.py
```

预期先列出上述四个训练 key，再列出 `boxes`、`labels`、`scores` 三个评估 key。命令用随机
权重构造 `fasterrcnn_mobilenet_v3_large_320_fpn`，只进行前向传播，不学习，也不发布成绩。

`examples/02_detection_batch.py` 展示两种模式都接收的列表容器：

```bash
uv run python examples/02_detection_batch.py
```

进入模型变换层前，预期形状仍为 `(3,16,20)` 与 `(3,12,24)`。模型内部可能缩放和填充，
但不会改变数据集的坐标约定。

## 从四项损失到一次参数更新

生产代码的一步优化是：

```text
model.train()
optimizer.zero_grad(set_to_none=True)
losses = model(images, targets)
loss_total = sum(losses.values())
loss_total.backward()
optimizer.step()
```

第 04 章会在准备好的数据上运行这条路径。`examples/04_minimal_training_loop.py` 用只含两项
损失的假检测器隔离优化机制，不要把它的损失字典误当成 Faster R-CNN 的四项契约。

## 常见失败边界

- 训练模式下不传标注：训练调用不完整。
- 评估模式下传标注后仍期待损失：评估模式返回预测结果。
- 传入单个堆叠张量而不是列表：破坏公开检测契约，原始尺寸也会变得含糊。
- 用标签 `0` 标记目标：它会被解释为背景。
- 把 RPN 目标性当成 VOC 分类：RPN 与具体类别无关。
- 把随机权重下的有限损失或正确键名当作学习质量：这里只验证了软件契约。

下一步进入[教程 04](04-training.zh-CN.md)，执行一次参数更新，并区分试运行、有界学习和
完整实验。
