# Object Detection Tutorial

[简体中文](README.zh-CN.md) | [Documentation](../README.md)

This tutorial accompanies one Kaggle training run and explains the data,
losses, metrics, and images you see along the way. You can submit the job first
and read the early chapters while it runs.

## Suggested order

| Chapter | What it explains | Do I need to run code? |
|---|---|---|
| [Learning path](learning-path.md) | How the whole project fits together | No |
| [00 - Detection basics](00-basics.md) | Images, boxes, classes, and variable-size batches | Optional small local examples |
| [01 - Environment](01-environment.md) | Why Kaggle is recommended and what to check locally | Needed when submitting to Kaggle |
| [02 - VOC data](02-data-and-boxes.md) | Splits, coordinates, and difficult objects | Kaggle prepares it automatically |
| [03 - Faster R-CNN](03-faster-rcnn.md) | RPN, ROI, training losses, and predictions | No |
| [04 - Training](04-training.md) | How to submit, read logs, and select the best epoch | Yes, on a Kaggle GPU |
| [05 - Evaluation and prediction](05-evaluation-and-inference.md) | How to read metrics, false positives, misses, and images | You can inspect the saved result |

## Run first or read first?

Both approaches work:

- To see a result quickly, submit the job with the
  [Kaggle guide](../guides/kaggle.md), then read the tutorial.
- To understand the model first, read Chapters 00, 02, and 03 before training.

The only complete training result published by this project comes from Kaggle
v7. Synthetic tensors, randomly initialized models, and CPU dry runs in the
tutorial isolate one idea; they are not model scores.

After your run finishes, compare its `metrics.csv`, `evaluation.json`, and
prediction images with the [saved Kaggle run](../recorded-run/README.md).
