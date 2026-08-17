# 示例

请按顺序运行示例：xyxy 框与类别、可变检测 batch、检测器损失字典、一次 fake 优化步骤，以及仅依赖 checkpoint 的预测。前四个不需要文件或网络，第五个只接受本地输入。

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
uv run python examples/05_checkpoint_prediction.py --checkpoint artifacts/run/best.pt --image image.jpg
```

参见 [English](README.md)。
