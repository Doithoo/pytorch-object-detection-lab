# Registered Model Reference

[Simplified Chinese](model-zoo.zh-CN.md) | [Using models](../guides/using-models.md)

This page is the version 0.1 registry contract, not a leaderboard. All three models are torchvision constructors maintained by this repository. There is no stable external model plugin API. The Faster R-CNN MobileNet recipe has one [recorded full-VOC run](../recorded-run/README.md); it does not rank the other models.

## Discovery

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

These commands read registry metadata only. They do not construct a model, inspect the weight cache, contact the network, or write artifacts. Unknown names fail and may suggest a close registered name.

## Registry entries

| Name | Family | Backbone/input ownership | Shipped recipe and comparison role |
|---|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | `two_stage` | MobileNet V3 Large with FPN; detector transform defaults to shorter edge 320 and longer-edge cap 640 | Default and tutorial baseline in `configs/learning_minimal.yaml`; teaches backbone/FPN, RPN, and ROI heads |
| `fasterrcnn_resnet50_fpn` | `two_stage` | ResNet-50 with FPN; detector defaults to shorter edge 800 and longer-edge cap 1333 | `configs/fasterrcnn_resnet50_fpn.yaml`; changes backbone while retaining Faster R-CNN, with more memory and compute than the MobileNet recipe |
| `ssdlite320_mobilenet_v3_large` | `one_stage` | MobileNet V3 Large; built-in transform resizes to the 320-pixel SSDLite recipe | `configs/ssdlite320_mobilenet_v3.yaml`; compares a one-stage family |

The roles above describe architecture and maintained recipes. They do not establish relative accuracy, throughput, convergence, hardware support, or universal suitability.

## Weight policies

Every entry supports exactly `none` and `imagenet1k_v1`.

| Policy | Full detector | Backbone | Cache/network behavior |
|---|---|---|---|
| `none` | `weights=None` | `weights_backbone=None` | offline construction path; random initialization |
| `imagenet1k_v1` | `weights=None` | pinned torchvision `IMAGENET1K_V1` enum for the registered backbone | uses existing expected cache file or lets torchvision download during construction |

The MobileNet models expect `torch.hub.get_dir()/checkpoints/mobilenet_v3_large-8738ca79.pth`; ResNet-50 expects `torch.hub.get_dir()/checkpoints/resnet50-0676ba61.pth`. Preflight only checks existence and prints a network notice when absent. It neither downloads nor validates arbitrary replacement files. Checkpoint evaluation and prediction always rebuild with `none`, then load saved model state.

## Maintained `model.params`

The registry accepts only these constructor keys. Values are parsed as YAML and passed to torchvision; the project does not add range validation, so upstream constructor/runtime errors remain the value-validation boundary. Misspelled or unmaintained keys fail before construction.

| Model(s) | Key | Upstream default in this constructor | Type/effect |
|---|---|---:|---|
| both Faster R-CNN entries | `min_size` | MobileNet 320; ResNet-50 800 | positive integer shorter-edge target owned by detector transform |
| both Faster R-CNN entries | `max_size` | MobileNet 640; ResNet-50 1333 | positive integer longer-edge cap after aspect-preserving resize |
| both Faster R-CNN entries | `box_score_thresh` | `0.05` | numeric ROI inference score threshold |
| SSDLite | `score_thresh` | `0.001` | numeric inference score threshold before NMS |
| SSDLite | `nms_thresh` | `0.55` | numeric IoU threshold for non-maximum suppression |
| SSDLite | `detections_per_img` | `300` | positive integer cap after NMS |

Example:

```yaml
model:
  name: fasterrcnn_mobilenet_v3_large_320_fpn
  weights: none
  expected_num_classes: 21
  params:
    min_size: 320
    max_size: 640
    box_score_thresh: 0.05
```

`weights`, `weights_backbone`, and `num_classes` are reserved and fail if placed in `model.params`. Any key not listed for the selected model also fails; use `detect model-info MODEL_NAME` to inspect the maintained surface.

## Shared input and mode contract

All models accept a list of RGB float tensors `[3,H,W]` in `[0,1]`; model-owned transforms normalize, resize, and batch internally. Training receives an aligned target list with zero-based continuous `xyxy` boxes and foreground labels. Train mode returns a nonempty scalar loss mapping; eval mode returns `boxes`, `labels`, and `scores` per image. Faster R-CNN's exact four losses are documented in [how Faster R-CNN works](../concepts/how-faster-rcnn-works.md).

VOC metadata supplies 20 foreground classes plus background, so `model.expected_num_classes` must be 21. To make a controlled choice, hold weight policy, manifest identity, sample limits, seed, optimizer, and epochs fixed, then compare validation artifacts as described in [experiments](../guides/experiments.md).
