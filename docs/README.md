# Documentation

[简体中文](README.zh-CN.md) | [Project home](../README.md)

You do not need to read these pages from top to bottom. Start with what you
want to do today.

## I want to train on Kaggle

Begin with the [Kaggle training guide](guides/kaggle.md). It covers account
setup, CLI authentication, submission, GPU checks, logs, and result downloads.
When you want the ideas behind the run, read:

1. [Object-detection basics](tutorial/00-basics.md)
2. [VOC data and bounding boxes](tutorial/02-data-and-boxes.md)
3. [Faster R-CNN](tutorial/03-faster-rcnn.md)
4. [Training](tutorial/04-training.md)
5. [Evaluation and prediction](tutorial/05-evaluation-and-inference.md)

The completed Kaggle T4 run and its real outputs are in the
[training record](recorded-run/README.md).

## I want to understand the code

- [Learning path](tutorial/learning-path.md): the complete map from boxes to a
  trained result.
- [Detection flow](concepts/detection-flow.md): how an image moves through the
  dataset, model, and evaluation code.
- [How Faster R-CNN works](concepts/how-faster-rcnn-works.md): RPN, ROI, and the
  loss terms.
- [Code tour](concepts/code-tour.md): where the CLI, data, model, training, and
  evaluation code live.
- [Configuration flow](concepts/configuration-flow.md): how YAML and command-line
  overrides combine.
- [Example programs](../examples/README.md): small programs you can run alone.

## I need a specific answer

| Question | Page |
|---|---|
| A Kaggle run failed | [Troubleshooting](guides/troubleshooting.md) |
| Which model should I choose? | [Using models](guides/using-models.md) |
| How can I modify a detector? | [Model modification example](guides/modifying-models.md) |
| What does a configuration field mean? | [Configuration reference](reference/config-reference.md) |
| How are VOC manifests and annotations stored? | [Dataset format](reference/dataset-format.md) |
| What do metrics and output files mean? | [Metrics reference](reference/metrics.md) |
| What is inside a checkpoint? | [Checkpoint format](reference/checkpoint-schema.md) |
| How do I use my own data? | [Custom data guide](guides/using-your-data.md) |
| How do I prepare COCO JSON data? | [COCO data guide](guides/using-coco.md) |
| How do I compare two runs? | [Experiment guide](guides/experiments.md) |

When adding a dataset or model, or changing internal behavior, continue to
[adding datasets](guides/adding-datasets.md), [adding models](guides/adding-models.md),
and the [architecture note](architecture/0001-reproducible-voc-detection-contracts.md).
