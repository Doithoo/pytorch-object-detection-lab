# 教程 02：认识 VOC 数据与边界框

[English](02-data-and-boxes.md) | [教程索引](README.zh-CN.md)

Pascal VOC 2007 是一个适合入门的目标检测数据集：规模不大、标注格式清楚，并包含 20 个
常见物体类别。本项目使用官方划分：

| 划分 | 图像数 | 用途 |
|---|---:|---|
| train | 2,501 | 更新模型参数 |
| valid | 2,510 | 比较轮次并选择 `best.pt` |
| test | 4,952 | 训练结束后的最终评估 |

Kaggle runner 会自动下载、解压并准备这些数据。第一次学习时不需要手动上传数据集。

## 一条标注包含什么

VOC 为每张图像提供一个 XML 文件。项目把它转换为 torchvision 使用的 target 字典，
其中最重要的字段是：

```text
boxes:      FloatTensor[N, 4]
labels:     Int64Tensor[N]
image_id:   Int64Tensor[1]
area:       FloatTensor[N]
iscrowd:    Int64Tensor[N]
difficult:  BoolTensor[N]
```

`N` 是这张图像中的目标数。没有普通目标时，`boxes` 的形状仍然是 `[0, 4]`，而不是一维
空张量。

## VOC 坐标怎样转换

VOC XML 使用从 1 开始、包含端点的坐标。项目只转换一次：

```text
(xmin - 1, ymin - 1, xmax, ymax)
```

例如 `(11, 21, 50, 70)` 会变成 `[10, 20, 50, 70]`。转换后宽度是 `40`，高度是
`50`，面积是 `2000`。后续数据增强或模型代码不要再次减 1，也不要额外加 1。

![图像、边界框和 target 字段示意图](../assets/detection-target-anatomy.png)

这是一张教学示意图，不是训练结果。绿色框表示普通目标，橙色虚线框表示 difficult 目标。

## difficult 目标

VOC 使用 `difficult=1` 标记难以可靠识别或定位的物体。本项目的处理方式是：

- 训练时不把 difficult 目标送入损失计算。
- 验证和测试时保留它们。
- 只匹配 difficult 目标的预测不会被当成普通误检。

这样可以避免用含糊标注训练模型，同时在评估和可视化中保留原始信息。

## Kaggle 中的数据准备

runner 会：

1. 从官方地址下载训练/验证和测试压缩包。
2. 检查官方 MD5。
3. 解压到 `/kaggle/working/data`。
4. 生成 `train.csv`、`valid.csv`、`test.csv` 和 `dataset.yaml`。
5. 在训练前读取一个 batch，确认图像和框能够进入模型。

日志出现下面两行时，数据下载已经完成：

```text
{"phase": "download_voc2007", "status": "started"}
{"phase": "download_voc2007", "status": "completed", ...}
```

准备后的数据标识会写入 `run.yaml` 和 checkpoint，用于确认评估时使用的是同一份数据。
初学阶段只需知道它相当于这次数据的“版本号”。

## 可选：在本地查看 VOC

如果想亲自打开图像和标注，在仓库根目录运行：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset_preview.png
```

`inspect-data` 输出图像尺寸、目标数、类别和框范围。预览图把标注画到真实图像上，适合检查
类别和框的位置。`--limit` 只控制这次查看多少张图，不会改变官方划分。

## 看到异常时先检查什么

- 框整体偏移一个像素：检查是否重复执行了 `xmin - 1` 和 `ymin - 1`。
- `boxes` 形状错误：空标注也必须是 `[0, 4]`。
- 训练图像没有目标：检查其中的对象是否全部标为 difficult。
- Kaggle 下载失败：确认任务 Internet 已开启，再查看官方主机是否暂时不可达。
- 数据数量不是 `2501 / 2510 / 4952`：确认使用的是完整官方 VOC 2007。

下一步阅读 [Faster R-CNN](03-faster-rcnn.zh-CN.md)，看这些图像和 target 怎样进入模型。
