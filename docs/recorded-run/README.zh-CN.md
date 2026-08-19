# Kaggle VOC 2007 训练记录

[English](README.md) | [Kaggle 训练指南](../guides/kaggle.zh-CN.md) | [参考配置](../../configs/reference_fasterrcnn.yaml)

这是项目发布的完整训练结果。任务于 2026-08-19 在 Kaggle Tesla T4 上成功完成，训练 26
轮，并使用验证集选出的第 18 轮模型评估官方 VOC 2007 测试集。

## 结果

| 项目 | 数值 |
|---|---:|
| 模型 | Faster R-CNN MobileNet V3 Large 320 FPN |
| Backbone 权重 | ImageNet1K V1 |
| 完成轮次 | 26 |
| 最佳验证轮次 | 18 |
| 最佳验证 `map_50_95` | 0.313245 |
| 测试图像 | 4,952 |
| 测试普通目标 / 预测 | 12,032 / 26,353 |
| 测试 `map_50_95` | **0.322312** |
| 测试 `map_50` / `map_75` | 0.609917 / 0.302681 |
| 测试 `mar_1` / `mar_10` / `mar_100` | 0.338981 / 0.413547 / 0.415008 |
| 训练 / 测试评估时间 | 3,025.660 秒 / 74.893 秒 |
| Kaggle 任务总时间 | 3,223.9 秒 |

AP/AR 由 torchmetrics 1.9.0 和 pycocotools 2.0.11 计算。这里的 `map_50_95` 是 COCO
风格的多个 IoU 阈值平均，不是历史 VOC 2007 的 11 点算法。数值是 0 到 1 的小数。

## 先看模型做了什么

![测试图像 000001 的标注与预测](evaluation/visualizations/summary.png)

绿色框是普通标注，橙色虚线框是 difficult 目标，蓝色框是模型预测。再打开两个真实失败
案例：

- [误检案例](evaluation/visualizations/false_positive-01-009040.png)
- [漏检案例](evaluation/visualizations/missed-01-006500.png)

单张图不能代表全部 4,952 张测试图像。它的作用是帮助你把汇总指标、CSV 记录和具体图像
对应起来。

## 可以直接查看的文件

| 文件 | 内容 |
|---|---|
| [`metrics.csv`](metrics.csv) | 26 轮训练 loss 和验证指标 |
| [`config.yaml`](config.yaml) | Kaggle 实际使用的完整配置 |
| [`run.yaml`](run.yaml) | 数据划分、设备、版本、随机种子和运行时间 |
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | 轮次、耗时、划分数量和未舍入测试指标 |
| [`evaluation/evaluation.json`](evaluation/evaluation.json) | 测试集汇总、阈值和后端版本 |
| [`evaluation/per_class.csv`](evaluation/per_class.csv) | 20 个 VOC 类别各自的指标 |
| [`evaluation/errors.csv`](evaluation/errors.csv) | ignored、定位问题、误检和漏检记录 |

建议先读 `metrics.csv` 中的 `epoch`、`loss_total` 和 `valid_map_50_95`，再看
`evaluation.json` 和图像。完整阅读方法见[评估教程](../tutorial/05-evaluation-and-inference.zh-CN.md)。

## 这次运行怎样完成

- Kaggle kernel：`yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7`
- 页面：<https://www.kaggle.com/code/yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7>
- 设备：`cuda:0`，Tesla T4。Kaggle 分配了两张，本项目使用一张。
- 数据：官方 VOC 2007，train / valid / test 为 2,501 / 2,510 / 4,952。
- 随机种子：42。
- 训练：26 轮，CUDA AMP，两个 data workers。
- 选模：每轮在 valid 上计算 `map_50_95`，保存最好轮次。
- 测试：全部训练结束后，用 `best.pt` 评估一次 test。

实际提交文件是 [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 和
[`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json)。runner 内嵌源码，不需要
附加 Dataset 或 `kagglehub`。

## 自己运行一次

把 metadata 的 `id` 改成自己的账户，然后：

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

请求 T4 或更新 GPU并开启 Internet。当前 Kaggle PyTorch 不支持 P100 的 `sm_60`，P100
会在训练开始时失败。完整步骤和已知错误见 [Kaggle 指南](../guides/kaggle.zh-CN.md)。

## 复现信息

以下细节用于确认结果来自哪一次运行，初次阅读可以跳过：

| 项目 | 记录值 |
|---|---|
| Python | 3.12.13 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| 数据标识 | `b9bdc2604322c07f26c9a0135a063c7702b0dfb261171401076cf6733cfdb5b7` |
| 内嵌源码大小 / SHA-256 | 157,993 字节 / `2186866a9b4b582e2c2c38128a178bd958ffaa5b0dcafbf6d4c55e4f39aca628` |
| 被评估 checkpoint SHA-256 | `826e2bb38b985945fbfbaf59587e06ecb9fc70501c5ce80f6d1e357b59b0826a` |

`run.yaml` 中的 `git_revision` 是 `null`，因为 runner 保存的是当时的源码快照，而不是一个
已经存在的 Git commit。精确源码仍嵌在 runner 中，并用上面的摘要标识。

仓库不提交 145 MB checkpoint、下载后的 VOC 数据和 5.6 MB 完整预测列表。需要模型时，
请从自己的 Kaggle 任务下载 `best.pt`。保留下来的配置、指标、错误 CSV 和三张图足以阅读
这次公开结果，但不表示该模型是 VOC 上的最佳模型。
