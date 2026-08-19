# 示例

[English](README.md) | [教程](../docs/tutorial/README.zh-CN.md)

建议按顺序运行。示例从纯张量检测契约，逐步进入真实 torchvision 模型契约、一次合成优化，再到基于 checkpoint 的预测。它们是教学探针，不是 benchmark 运行。

## 渐进路线

| 示例 | 用于理解 | 前提 | 网络行为 | 预期输出或产物 |
|---|---|---|---|---|
| `01_boxes_and_labels.py` | xyxy 坐标、整数类别与面积 | 已安装 PyTorch | 离线 | stdout 输出两个框、类别和计算面积 |
| `02_detection_batch.py` | 检测 batch 为什么由不同尺寸图像与 targets 的列表组成 | 已安装本项目 | 离线 | 两种图像形状，以及 target 数量 `1` 和 `2` |
| `03_detector_losses.py` | 训练模式返回的具名标量 loss 字典 | 已安装本项目 | 离线 | tiny fake detector 的分类 loss、框 loss 与总和 |
| `03_model_contract.py` | 真实 torchvision 模型的 train/eval API 边界 | 比纯张量示例更多的 CPU 时间与内存 | 离线，因为使用 `weights: none` 构造模型 | 先输出训练 loss keys，再输出 boxes、labels、scores 等评估 keys |
| `04_minimal_training_loop.py` | 清空梯度、反向传播与一次 SGD 更新 | 已安装本项目 | 离线 | 一次更新前后的合成参数值 |
| `05_checkpoint_prediction.py` | 从 checkpoint 恢复模型与类别 | 本地自包含 checkpoint 和图像 | 离线；恢复时强制使用 `weights: none` | 指定输出目录中的 `<stem>.json` 与 `<stem>.png` |

## 运行命令

前五条命令不需要数据集或输入文件：

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/03_model_contract.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

最后一条使用已经完成的训练产物与本地图像：

```bash
uv run python examples/05_checkpoint_prediction.py --checkpoint artifacts/run/best.pt --image image.jpg --output-dir artifacts/example_prediction
```

只需要理解 API 契约时，先运行 `03_detector_losses.py`，它快速且完全合成。`03_model_contract.py` 会构造 torchvision 提供的正式检测器，因此明显更慢，但它仍然不进行训练，也不会发布成绩。

## 证据边界

示例 01-04 使用合成值，只验证局部数据、模型模式和优化契约。示例 05 验证兼容 checkpoint 能够驱动本地预测。所有示例都不会下载 VOC、在完整数据集上训练或建立完整 VOC 成绩。需要有界的集成流程时，请继续阅读[教程](../docs/tutorial/README.zh-CN.md)。
