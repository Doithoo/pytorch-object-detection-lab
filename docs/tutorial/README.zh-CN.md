# 教程

[English](README.md) | [文档索引](../README.zh-CN.md)

第一次端到端使用请走这条路线。各章会为下一章引入必要契约，因此建议按顺序阅读。实际操作顺序严格为 `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`。

## 开始前

安装 Python 3.10-3.12 与 uv，克隆仓库，然后运行 `uv sync --locked --extra dev`。第 00、01、03 章使用合成输入，不需要 VOC 文件。进入第 02 章前，请先按仓库的[主数据准备流程](../../README.zh-CN.md)下载、校验并生成 manifests；本章随后解释 VOC 坐标和 difficult 目标，并预览已经准备好的划分。默认学习配置使用 `weights: none`，模型路径不会下载权重。

## 章节地图

| 章节 | 何时阅读 | 前提 | 预期输出或结论 |
|---|---|---|---|
| [学习路径](learning-path.zh-CN.md) | 希望先掌握全流程 | 环境已经安装 | CLI 版本，以及七个阶段的整体地图 |
| [00 - 检测基础](00-basics.zh-CN.md) | 第一次接触框、类别、空 target 或不同尺寸图像 | 基础 Python 与张量索引 | 输出 xyxy 框、整数类别、面积和不同 batch 形状 |
| [01 - 环境](01-environment.zh-CN.md) | 需要确认锁定环境与离线权重策略 | 仓库依赖已安装 | 解析后的学习 YAML，其中包含 `weights: none` 和样本上限 |
| [02 - VOC 数据与框](02-data-and-boxes.zh-CN.md) | 已有准备好的数据，需要理解坐标和 difficult 目标 | 主数据准备流程生成的 manifests 与对应源图像 | 坐标转换说明，以及显示普通框和 difficult 框的 `artifacts/dataset_preview.png` |
| [03 - Faster R-CNN](03-faster-rcnn.zh-CN.md) | 需要理解训练 losses 与评估 predictions 的差别 | 已理解 00-02 章概念 | 合成示例输出的具名 detector losses；不会产生已学习 checkpoint |
| [04 - 训练](04-training.zh-CN.md) | 数据和模型契约已经清楚 | 本地 manifests 与源图像 | 一次更新的 dry run 诊断，以及后续有界训练目录中的配置、来源、指标与 checkpoints |
| [05 - 评估与推理](05-evaluation-and-inference.zh-CN.md) | 已有 checkpoint，需要报告或预测 | 评估需要匹配的准备数据；预测只需要本地图像 | 评估 JSON/CSV/证据图，或预测 JSON/PNG |

## 流程检查点

训练前同时检查机器可读摘要与渲染样本：

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --limit 4
```

第一条命令输出数量、类别频次、框范围与 difficult 目标信息，第二条写入 `artifacts/dataset_preview.png`。它们都不能证明优化过程有效。

然后执行一次参数更新来验证训练流程：

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

预期输出包括图像形状、target 数量、值均为有限数的各项具名 loss，以及 `dry-run OK`。dry run 不写 checkpoint。使用同一配置正常训练时，train/valid/test 分别最多读取 32/16/16 个样本并生成学习产物；这不是完整 VOC benchmark。

## 完成教程代表什么

完成教程代表能够沿全流程追踪数据和产物。合成示例证明局部契约；有界学习运行证明集成路径能够执行和更新参数。[完整 VOC 实测运行](../recorded-run/README.zh-CN.md)展示更高证据层级：精确来源、验证集选择、完整划分范围、测试指标、checkpoint 哈希、耗时和真实失败案例图。
