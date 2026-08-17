# Examples

Run the examples in order. They cover xyxy boxes and labels, variable detection batches, detector loss dictionaries, one fake optimization step, and checkpoint-only prediction. The first four require no files or network; the fifth accepts only local inputs.

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
uv run python examples/05_checkpoint_prediction.py --checkpoint artifacts/run/best.pt --image image.jpg
```

See [中文说明](README.zh-CN.md).
