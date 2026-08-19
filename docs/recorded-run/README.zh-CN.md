# 完整 VOC 运行记录

[English](README.md) | [参考配置](../../configs/reference_fasterrcnn.yaml) | [Kaggle 指南](../guides/kaggle.zh-CN.md)

本目录记录一次已经完成、证据完整的 Pascal VOC 2007 运行。它是可复现的项目实测
结果，不代表该配方是通用 torchvision benchmark，也不宣称它是 VOC 上的最佳检测器。

## 结果

| 字段 | 实测值 |
|---|---:|
| 模型 | Faster R-CNN MobileNet V3 Large 320 FPN |
| Backbone 权重 | ImageNet1K V1 |
| 完成轮次 | 26 |
| 最佳验证轮次 | 18 |
| 最佳验证 `map_50_95` | 0.313245 |
| 测试图像数 | 4,952 |
| 测试目标数 / 预测数 | 12,032 / 26,353 |
| 测试 `map_50_95` | **0.322312** |
| 测试 `map_50` / `map_75` | 0.609917 / 0.302681 |
| 测试 `mar_1` / `mar_10` / `mar_100` | 0.338981 / 0.413547 / 0.415008 |
| 训练 / 测试评估耗时 | 3,025.660 秒 / 74.893 秒 |
| Kaggle notebook 总耗时 | 3,223.9 秒 |

AP/AR 来自 torchmetrics 1.9.0 与 pycocotools 2.0.11，是本项目实现的 COCO
风格 IoU 阈值扫描，不是历史 VOC 2007 的 11 点评估器。指标是无单位小数。

![测试图像 000001 的目标与预测](evaluation/visualizations/summary.png)

绿色框是普通目标，橙色虚线框是困难目标，蓝色框是预测。还应查看记录的
[误检案例](evaluation/visualizations/false_positive-01-009040.png)和
[漏检案例](evaluation/visualizations/missed-01-006500.png)，不要把首张汇总图当作整个
测试集的代表。

## 运行标识

| 字段 | 实测值 |
|---|---|
| 完成日期 | 2026-08-19 |
| Kaggle kernel | `yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7` |
| Kernel URL | <https://www.kaggle.com/code/yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7> |
| 设备 | `cuda:0`、Tesla T4；分配了两张 GPU，实际使用一张 |
| Python | 3.12.13 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| 随机种子 | 42 |
| 数据集标识 | `b9bdc2604322c07f26c9a0135a063c7702b0dfb261171401076cf6733cfdb5b7` |
| train / valid / test 图像数 | 2,501 / 2,510 / 4,952 |
| 嵌入源码压缩包 | 157,993 字节；SHA-256 `2186866a9b4b582e2c2c38128a178bd958ffaa5b0dcafbf6d4c55e4f39aca628` |
| 被评估 checkpoint | SHA-256 `826e2bb38b985945fbfbaf59587e06ecb9fc70501c5ce80f6d1e357b59b0826a` |

`run.yaml` 中的 `git_revision` 是 `null`，因为 Kaggle runner 嵌入的是当时未提交的
项目快照，而不是某个 Git commit。精确的 157,993 字节源码压缩包仍嵌在
[`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 中，其摘要记录在上表。这样既保存了
实际执行源码，也不会假装存在一个并未创建的 commit。

官方划分哈希保存在 [`run.yaml`](run.yaml) 中。解析后的 [`config.yaml`](config.yaml)
记录 Kaggle 覆盖项：CUDA、AMP、两个数据 worker 与 `/kaggle/working` 路径。模型只按
验证集 `map_50_95` 选择；第 26 轮结束后才评估保留的测试集。

## 保存的证据

- [`metrics.csv`](metrics.csv)：完整 26 轮训练与验证记录。
- [`kaggle-run-summary.json`](kaggle-run-summary.json)：划分数量、耗时和未舍入测试指标。
- [`evaluation/evaluation.json`](evaluation/evaluation.json)：舍入后的测试指标、阈值、
  后端版本、数据集标识和 checkpoint 哈希。
- [`evaluation/per_class.csv`](evaluation/per_class.csv)：全部 20 个 VOC 类别。
- [`evaluation/errors.csv`](evaluation/errors.csv)：`ignored`、定位错误、误检和漏检记录。
- [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 与
  [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json)：实际提交的 runner 和
  Kaggle 机器配置。

仓库不提交 145 MB checkpoint、下载的 VOC 数据或 5.6 MB 全量预测列表。上面的
checkpoint 哈希把报告与被评估文件绑定。保留的三张图只是代表性证据，不是经过挑选的
精度宣传。

## 复现或下载

实际 runner 是自包含脚本，但需要 Kaggle 网络下载官方 VOC 压缩包和 ImageNet
backbone 权重。推送自己的副本前，把 metadata 中的 `id` 改成自己的 Kaggle 账户：

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7
kaggle kernels output yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7 --file-pattern 'artifacts/.*' -p kaggle-output
```

应请求 T4 或更新 GPU。当前 Kaggle PyTorch 构建不包含 P100 的 `sm_60` kernel，P100
会在训练前失败。runner 有意只使用一张 GPU，不宣称多 GPU 加速。简短操作流程见
[Kaggle 指南](../guides/kaggle.zh-CN.md)。
