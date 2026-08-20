# 文档导航

[English](README.md) | [项目首页](../README.zh-CN.md)

第一次使用时，不需要从头读到尾。先选择你现在想做的事。

## 我想在 Kaggle 训练

从 [Kaggle 训练指南](guides/kaggle.zh-CN.md)开始。它会带你完成账户准备、CLI 认证、
提交、GPU 检查、日志判断和结果下载。训练过程中遇到概念，再按顺序阅读：

1. [目标检测基础](tutorial/00-basics.zh-CN.md)
2. [VOC 数据与边界框](tutorial/02-data-and-boxes.zh-CN.md)
3. [Faster R-CNN](tutorial/03-faster-rcnn.zh-CN.md)
4. [训练](tutorial/04-training.zh-CN.md)
5. [评估与预测](tutorial/05-evaluation-and-inference.zh-CN.md)

已经完成的 Kaggle T4 运行及其真实输出保存在
[训练记录](recorded-run/README.zh-CN.md)中。

## 我想理解项目代码

- [学习路线](tutorial/learning-path.zh-CN.md)：从边界框到训练结果的完整地图。
- [检测流程](concepts/detection-flow.zh-CN.md)：一张图像怎样经过数据集、模型和评估。
- [Faster R-CNN 原理](concepts/how-faster-rcnn-works.zh-CN.md)：RPN、ROI 和损失项。
- [代码导览](concepts/code-tour.zh-CN.md)：CLI、数据、模型、训练和评估代码在哪里。
- [配置流程](concepts/configuration-flow.zh-CN.md)：YAML 和命令行覆盖怎样组合。
- [示例程序](../examples/README.zh-CN.md)：可以单独运行的小例子。

## 我想查一个具体问题

| 问题 | 文档 |
|---|---|
| Kaggle 运行失败 | [排错指南](guides/troubleshooting.zh-CN.md) |
| 应该选哪个模型 | [使用模型](guides/using-models.zh-CN.md) |
| 如何修改 detector | [模型修改示例](guides/modifying-models.zh-CN.md) |
| 配置字段是什么意思 | [配置参考](reference/config-reference.zh-CN.md) |
| VOC 清单和标注格式 | [数据格式](reference/dataset-format.zh-CN.md) |
| 指标和输出文件 | [指标参考](reference/metrics.zh-CN.md) |
| checkpoint 中有什么 | [checkpoint 结构](reference/checkpoint-schema.zh-CN.md) |
| 使用自己的数据 | [自定义数据指南](guides/using-your-data.zh-CN.md) |
| 准备 COCO JSON 数据 | [COCO 数据指南](guides/using-coco.zh-CN.md) |
| 为 YOLO 导出数据 | [YOLO 导出指南](guides/using-yolo.zh-CN.md) |
| 比较两次运行 | [实验指南](guides/experiments.zh-CN.md) |

添加数据集、模型或修改内部行为时，再阅读[添加数据集](guides/adding-datasets.zh-CN.md)、
[添加模型](guides/adding-models.zh-CN.md)和[架构说明](architecture/0001-reproducible-voc-detection-contracts.zh-CN.md)。
