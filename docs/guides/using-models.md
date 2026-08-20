# Choose a Model

[简体中文](using-models.zh-CN.md) | [Model reference](../reference/model-zoo.md)

The project includes five torchvision detectors plus explicit external factories. The Faster R-CNN MobileNet entry has a recorded Kaggle run; the other entries provide architecture comparisons under the same project contracts.

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

These commands only display model information. They do not download weights or
start training.

## Model characteristics

| Model | Character | Suggestion |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | Compact two-stage model with a completed Kaggle VOC run | Recorded reference implementation |
| `fasterrcnn_resnet50_fpn` | Two-stage model with a larger backbone and more compute | Isolates backbone capacity within Faster R-CNN |
| `retinanet_resnet50_fpn` | Anchor-based one-stage model with focal loss | Shows dense anchors and class-imbalance handling |
| `fcos_resnet50_fpn` | Anchor-free one-stage model with centerness | Shows location-based prediction without predefined anchors |
| `ssdlite320_mobilenet_v3_large` | Compact one-stage model with fixed 320 input | Shows a mobile-oriented dense detector |

The project has not published a full five-model comparison under one training
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
prediction filtering. Change one relevant parameter at a time when the comparison depends on that parameter.

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
