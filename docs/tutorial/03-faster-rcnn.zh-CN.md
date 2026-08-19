# 教程 03：Faster R-CNN 怎样找到物体

[English](03-faster-rcnn.md) | [教程索引](README.zh-CN.md)

Faster R-CNN 是一个两阶段检测器：先找“可能有物体的区域”，再判断每个区域是什么并
调整边界框。理解这两步，就能看懂训练日志中的四项 loss。

## 输入为什么是列表

一批图像可以有不同尺寸，因此调用模型时使用：

```text
images  = [Tensor[3, H1, W1], Tensor[3, H2, W2], ...]
targets = [{boxes, labels, ...}, {boxes, labels, ...}, ...]
```

torchvision 在模型内部缩放、归一化和填充图像，并在输出时把框映射回原图尺寸。调用者不
需要先把所有图片拉伸成同样大小。

## 第一步：RPN 提出候选区域

backbone 和 FPN 先把图像变成多个尺度的特征图。Region Proposal Network（RPN）在这些
特征上寻找可能包含物体的区域。它只判断“这里像不像物体”，还不区分 person、dog 或
其他 VOC 类别。

RPN 对应两项训练 loss：

- `loss_objectness`：候选区域是物体还是背景。
- `loss_rpn_box_reg`：候选框需要怎样移动和缩放。

## 第二步：ROI head 分类并调整框

ROI Align 从特征图中取出每个候选区域的特征。ROI head 判断具体类别，并进一步调整
边界框。它对应另外两项 loss：

- `loss_classifier`：背景和具体物体类别的分类。
- `loss_box_reg`：最终边界框的位置调整。

项目把四项 loss 相加为 `loss_total`，用于反向传播和日志记录。不同 batch 的数值会变化，
单独一个较小的 loss 不能直接说明模型更好。

## 训练和预测时返回不同内容

| 状态 | 调用方式 | 返回内容 |
|---|---|---|
| 训练 | `model(images, targets)` | 四项 loss |
| 评估/预测 | `model(images)` | 每张图的 `boxes`、`labels`、`scores` |

可以用随机权重模型查看这种区别：

```bash
uv run python examples/03_model_contract.py
```

这个例子只做前向传播，不训练，也不产生模型成绩。它先打印训练 loss 的名称，再打印预测
字段。

不同尺寸 batch 的小例子是：

```bash
uv run python examples/02_detection_batch.py
```

## 一次参数更新发生了什么

训练代码的核心顺序是：

```text
清空旧梯度
-> 计算四项 loss
-> 求和得到 loss_total
-> 反向传播
-> 优化器更新参数
```

Kaggle runner 会先用一个 batch 完成 dry run，确认这条路径可以执行，再开始 26 轮训练。
日志中的 loss 来自 torchvision 模型，不是项目手工编出的评分。

## 容易混淆的地方

- 目标类别从 `1` 开始；`0` 留给背景。
- RPN 只提出区域，不预测最终 VOC 类别。
- 训练需要 targets，预测不需要 targets。
- 随机权重模型能返回有限 loss，不代表它已经学会检测。
- `scores` 是预测置信度，不是 IoU，也不是 mAP。

下一步进入[训练教程](04-training.zh-CN.md)，把这些 loss 与 Kaggle 日志和
`metrics.csv` 对应起来。
