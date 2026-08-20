# Modify a Detection Model

[简体中文](modifying-models.zh-CN.md) | [External factory reference](adding-models.md)

This page uses a small repository example to show which parts of a detector can
change while the data, training, checkpoint, and evaluation contracts stay the
same. It is a reference implementation rather than a performance claim.

## Example files

- `examples/extensions/custom_detector.py`: importable model factory.
- `configs/custom_detector_example.yaml`: resolved settings for that factory.

Inspect the configuration and construct one offline update:

```bash
uv run detect show-config --config configs/custom_detector_example.yaml
uv run detect train --config configs/custom_detector_example.yaml --dry-run --device cpu
```

The example builds Faster R-CNN from reusable torchvision components:

```text
MobileNet V3 Small features
-> explicit out_channels contract
-> custom AnchorGenerator
-> MultiScaleRoIAlign
-> FasterRCNN
```

## What changes

`width_mult` changes backbone capacity. Anchor sizes and aspect ratios change
which object shapes the RPN represents directly. `min_size` and `max_size`
change model-owned resizing. The ROI pooling and Faster R-CNN heads remain
upstream implementations.

The factory receives `num_classes` from prepared dataset metadata. It therefore
works with the built-in VOC data, custom VOC-shaped classes, and COCO JSON
without hardcoding label count.

## What stays fixed

The model still accepts `list[Tensor[3,H,W]]` and aligned targets in train mode,
returns a scalar loss mapping, and returns `boxes`, `labels`, and `scores` in
eval mode. That boundary lets the existing trainer, safe checkpoint loader,
metrics, error analysis, and predictor work without model-specific branches.

## Useful modifications

- Replace `mobilenet_v3_small(...).features` with another feature extractor and
  set its output-channel count accurately.
- Change anchor sizes after inspecting object-size statistics.
- Change aspect ratios when the data contains consistently narrow or wide
  objects.
- Replace the ROI head when the output classes or representation need a
  different design.
- Register a stable built-in name when a factory becomes a maintained project
  option; keep experimental factories explicit.

A dry run proves only that one forward/backward/update succeeds. Meaningful
claims require a fixed manifest, comparable settings, validation metrics,
runtime metadata, and saved failure examples.
