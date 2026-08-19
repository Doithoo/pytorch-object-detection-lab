# Choose a Model

[简体中文](using-models.zh-CN.md) | [Model reference](../reference/model-zoo.md)

The project includes three torchvision detectors. Use the Kaggle-tested Faster
R-CNN MobileNet for a first run, then compare other models after the workflow is
familiar.

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

These commands only display model information. They do not download weights or
start training.

## Choosing among the three models

| Model | Character | Suggestion |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | Compact two-stage model with a completed Kaggle VOC run | Use for the first run |
| `fasterrcnn_resnet50_fpn` | Two-stage model with a larger backbone and more compute | Compare when studying backbone effects |
| `ssdlite320_mobilenet_v3_large` | One-stage model with fixed 320 input | Use when comparing one-stage and two-stage detection |

The project has not published a full three-model comparison under one training
budget, so this table is not a speed or accuracy ranking.

## Weight settings

- `weights: imagenet1k_v1`: the detection head starts randomly while the
  backbone uses ImageNet pretrained weights. Kaggle reference training uses
  this setting and downloads the weight once.
- `weights: none`: detector and backbone both start randomly, useful for
  offline examples and dry runs.

`none` does not mean data is unnecessary; it only prevents a pretrained-weight
download during model construction.

## Inspect final parameters first

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

`min_size`, `max_size`, and `box_score_thresh` affect internal resizing and
prediction filtering. Keep project defaults for the first run rather than
changing several parameters together.

## Make a fair comparison

Keep data splits, weight policy, seed, optimizer, epochs, and sample limits
fixed, and change only `model.name`. Use a dry run first, then give both real
runs the same Kaggle GPU budget.

```bash
uv run detect train --config configs/learning_minimal.yaml --set model.name ssdlite320_mobilenet_v3_large --dry-run --device cpu
```

A dry run only checks one update. A complete comparison also needs validation
metrics, per-class results, runtime, and error images. See
[comparing training runs](experiments.md).
