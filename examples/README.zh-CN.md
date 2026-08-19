# 小型示例

[English](README.md) | [教程](../docs/tutorial/README.zh-CN.md)

这些程序一次只展示一个概念，适合在本地 CPU 上运行。它们使用合成数据或已有 checkpoint，
不会训练完整 VOC 模型，也不会产生可比较的模型成绩。

| 示例 | 用来理解什么 | 预期输出 |
|---|---|---|
| `01_boxes_and_labels.py` | xyxy 坐标、类别和面积 | 两个框、类别和面积 |
| `02_detection_batch.py` | 为什么不同尺寸图像保留为列表 | 两种图像形状和目标数量 |
| `03_detector_losses.py` | 训练模式返回 loss 字典 | 合成分类 loss、框 loss 和总和 |
| `03_model_contract.py` | 真实 torchvision 模型的训练/预测返回值 | loss 字段和预测字段名称 |
| `04_minimal_training_loop.py` | 梯度怎样更新一个参数 | 更新前后的参数值 |
| `05_checkpoint_prediction.py` | 怎样从 checkpoint 预测图像 | JSON 和标注 PNG |

前五个示例不需要 VOC：

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/03_model_contract.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

`03_model_contract.py` 会构造真实 torchvision 检测器，因此比其他合成示例慢，但仍不
训练。文件名中的 `contract` 是历史名称；示例实际展示的是模型在 train/eval 两种状态下
接收和返回什么。

最后一个示例需要已经下载的 checkpoint 和图片：

```bash
uv run python examples/05_checkpoint_prediction.py --checkpoint kaggle-output/reference-fasterrcnn/best.pt --image image.jpg --output-dir artifacts/example-prediction
```

想完成真实训练，请使用 [Kaggle 指南](../docs/guides/kaggle.zh-CN.md)。
