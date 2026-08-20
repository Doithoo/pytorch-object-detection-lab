# 目标检测教程

[English](README.md) | [文档导航](../README.zh-CN.md)

这套教程陪你完成一次 Kaggle 训练，并解释运行过程中看到的数据、loss、指标和图像。可以
先提交任务，再利用等待时间阅读前几章。

## 建议顺序

| 章节 | 你会弄清楚什么 | 是否需要运行代码 |
|---|---|---|
| [学习路线](learning-path.zh-CN.md) | 整个项目怎样串起来 | 否 |
| [00 - 检测基础](00-basics.zh-CN.md) | 图像、框、类别和不同尺寸 batch | 可选，小型本地示例 |
| [01 - 环境](01-environment.zh-CN.md) | 为什么推荐 Kaggle，以及本地能检查什么 | 提交 Kaggle 时需要 |
| [02 - VOC 数据](02-data-and-boxes.zh-CN.md) | 数据划分、坐标和 difficult 目标 | Kaggle 自动准备 |
| [03 - Faster R-CNN](03-faster-rcnn.zh-CN.md) | RPN、ROI、训练 loss 和预测 | 否 |
| [04 - 训练](04-training.zh-CN.md) | 怎样提交、看日志和选择最佳轮次 | 是，使用 Kaggle GPU |
| [05 - 评估与预测](05-evaluation-and-inference.zh-CN.md) | 怎样读指标、误检、漏检和预测图 | 可直接查看已保存结果 |

## 先运行还是先阅读？

两种方式都可以：

- 想尽快看到结果：先按 [Kaggle 指南](../guides/kaggle.zh-CN.md)提交任务，再读教程。
- 想先理解模型：先读 00、02、03 章，再提交训练。

项目唯一发布的完整训练结果来自已完成的 Kaggle 训练。教程中的合成张量、随机权重模型和
CPU dry run 都是用来观察某一步，不是模型成绩。

训练完成后，把自己的 `metrics.csv`、`evaluation.json` 和预测图与
[已保存的 Kaggle 运行](../recorded-run/README.zh-CN.md)逐项对照即可。
