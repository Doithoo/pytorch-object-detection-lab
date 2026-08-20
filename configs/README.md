# Training Configurations

[简体中文](README.zh-CN.md) | [Configuration field reference](../docs/reference/config-reference.md)

Values are combined in this order: defaults, YAML, then `--set KEY VALUE`.
Inspect the final values before training:

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

This command only prints configuration. It does not load data, construct a
model, or start training.

## Included configurations

| File | Use | Weights and network access |
|---|---|---|
| `reference_fasterrcnn.yaml` | Main Kaggle training: Faster R-CNN MobileNet V3, 26 epochs, full VOC | `imagenet1k_v1`; requires a download or existing cache |
| `learning_minimal.yaml` | Local dry run or small-sample code check | `none`; random initialization and no weight download |
| `custom_detector_example.yaml` | Import the repository's custom backbone/anchor factory example | `none`; no weight download by default |
| `fasterrcnn_resnet50_fpn.yaml` | Try a larger Faster R-CNN backbone | `none`; no weight download by default |
| `retinanet_resnet50_fpn.yaml` | Inspect an anchor-based one-stage detector with focal loss | `none`; no weight download by default |
| `fcos_resnet50_fpn.yaml` | Inspect an anchor-free one-stage detector | `none`; no weight download by default |
| `ssdlite320_mobilenet_v3.yaml` | Try the one-stage SSDLite model | `none`; no weight download by default |

The complete training result published by the project comes only from the
completed Kaggle run of `reference_fasterrcnn.yaml`. No full VOC result is
published for the other configurations.

## Reference commands

The recorded Kaggle run loads `reference_fasterrcnn.yaml` and changes the device, AMP, worker count, and paths for Kaggle. See the [Kaggle guide](../docs/guides/kaggle.md).

To check one local data and model update, use:

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

When comparing models, give each run a different name and keep data, seed,
epochs, and optimizer fixed:

```bash
uv run detect train --config configs/ssdlite320_mobilenet_v3.yaml --set run_name ssdlite-check --dry-run --device cpu
```

A dry run saves no checkpoint and does not mean the model finished training.

## What a run saves

A normal training directory contains `config.yaml`, `run.yaml`, `metrics.csv`,
`best.pt`, and `last.pt`. Keep `config.yaml`: it records the actual result after
defaults, YAML, and command-line overrides are combined.

Compare two compatible runs with:

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

Use validation to compare settings and evaluate test only after choices are
finished.
