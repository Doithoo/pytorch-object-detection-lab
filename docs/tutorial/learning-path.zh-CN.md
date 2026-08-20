# 从边界框到 Kaggle 训练

[English](learning-path.md) | [教程索引](README.zh-CN.md)

这条路线适合会写基本 Python、接触过 Tensor 和梯度，但还没有完整训练过目标检测模型的
读者。你不需要先准备本地 GPU。Kaggle runner 会自动完成：

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

## 1. 先知道模型要做什么

目标检测不只回答“图中有什么”，还要给出物体的位置。每个预测通常包含：

- `boxes`：形状为 `[N, 4]` 的 xyxy 边界框。
- `labels`：每个框的类别编号。
- `scores`：模型对每个预测的置信度。

阅读[检测基础](00-basics.zh-CN.md)后，你应该能看懂一条标注，不需要手算大量公式。

## 2. 认识这次训练的数据

项目使用 Pascal VOC 2007：2,501 张训练图像、2,510 张验证图像和 4,952 张测试图像，
包含 20 个物体类别。Kaggle runner 会下载官方压缩包并生成训练所需的清单。

[VOC 数据章节](02-data-and-boxes.zh-CN.md)会解释坐标为什么需要从 XML 转换，以及
`difficult` 目标为什么不会像普通目标那样计入评估。第一次阅读时理解含义即可，不必先
研究清单哈希和内部写入方式。

## 3. 看懂 Faster R-CNN 的两种输出

训练时，模型接收图像和标注，返回分类、框回归、RPN 分类和 RPN 框回归四项 loss。
评估时，模型只接收图像，返回框、类别和分数。

[Faster R-CNN 章节](03-faster-rcnn.zh-CN.md)会把 RPN 和 ROI head 串起来。先抓住“候选
区域”和“进一步分类定位”两个阶段，再看细节。

## 4. 在 Kaggle 提交训练

按 [Kaggle 指南](../guides/kaggle.zh-CN.md)完成四件事：

1. 安装 Kaggle CLI 并登录。
2. 把 kernel metadata 中的账户名改成自己的。
3. 提交 runner，并在网页确认 T4 或更新 GPU、Internet 已开启。
4. 等待状态变为 `COMPLETE`，然后只下载 `artifacts/.*`。

训练约需 50-60 分钟。日志持续输出心跳和 epoch 信息时，不需要刷新配置或重新提交。

## 5. 读懂训练过程

打开 `metrics.csv`，先看这些列：

- `epoch`：当前轮次。
- `loss_total`：训练 batch 上各项 loss 的总和。
- `valid_map_50_95`：验证集上的主要选择指标。
- `valid_map_50`：IoU 0.5 时的验证指标。

loss 和 mAP 衡量的不是同一件事，不要求它们同步变化。项目用验证集
`map_50_95` 选择 `best.pt`，而不是简单使用最后一轮。已保存的运行在第 18 轮取得最佳
验证结果，训练继续完成到第 26 轮。

## 6. 查看真实评估结果

进入 `evaluation/`：

- `evaluation.json` 给出测试集汇总。
- `per_class.csv` 展示 20 个类别的差异。
- `errors.csv` 记录误检、漏检和定位问题。
- `visualizations/` 让你直接看模型在哪里做对或做错。

可以先用[已保存的结果](../recorded-run/README.zh-CN.md)练习阅读，再看自己的文件。项目
发布的 Kaggle 测试结果是 `mAP@0.5:0.95 = 0.322312`、
`mAP@0.5 = 0.609917`。

## 7. 再决定下一步

完成第一次运行后，再选择一个方向：

- 想理解训练代码：阅读[训练章节](04-training.zh-CN.md)和[代码导览](../concepts/code-tour.zh-CN.md)。
- 想分析模型：阅读[评估与预测](05-evaluation-and-inference.zh-CN.md)。
- 想换模型：阅读[模型选择](../guides/using-models.zh-CN.md)。
- 想使用自己的数据：阅读[自定义数据指南](../guides/using-your-data.zh-CN.md)。
- 已有本地 GPU：使用训练章节末尾的本地命令。

一次只改变一个主要设置，并保留原始 `config.yaml` 和 `metrics.csv`，会比同时尝试很多
参数更容易看懂结果。
