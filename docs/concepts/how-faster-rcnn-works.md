# How Faster R-CNN Works in This Project

[Simplified Chinese](how-faster-rcnn-works.zh-CN.md) | [Tutorial chapter](../tutorial/03-faster-rcnn.md)

This page is for readers who know tensors but want to assign responsibilities inside the maintained Faster R-CNN models. It explains the public torchvision contract, not a reimplementation.

## Model-owned image transform

The caller supplies `list[float Tensor[3,Hi,Wi]]` in RGB `[0,1]`. Torchvision's generalized detector transform normalizes each image, resizes it with the registered `min_size`/`max_size` policy, pads the resized tensors into an internal batch, and remembers per-image sizes. Final boxes are mapped back to the caller's original coordinate system. The dataset must not preapply another detector resize/normalization policy.

## Backbone and FPN

The backbone converts pixels into spatial feature tensors. MobileNet V3 Large or ResNet-50 determines the feature extractor. The Feature Pyramid Network combines deep semantic features with finer spatial detail and exposes several resolutions with a common channel interface.

```text
padded images [B,3,H,W]
  -> backbone stages
  -> FPN lateral/top-down fusion
  -> feature maps {level: [B,C,Hk,Wk]}
```

These maps are shared evidence for proposals and ROI classification. They are not yet boxes or VOC class predictions.

## Region Proposal Network

The RPN evaluates anchors across FPN levels. It predicts class-agnostic objectness and box adjustments, decodes and clips proposals, ranks them, and suppresses duplicates before passing a variable number of regions per image to the second stage.

Its two training losses are:

| Key | Responsibility |
|---|---|
| `loss_objectness` | classify sampled anchors as object or background |
| `loss_rpn_box_reg` | regress positive anchors toward matched target boxes |

Objectness does not choose `person`, `dog`, or another VOC class. Calling an RPN proposal a final class prediction assigns responsibility to the wrong stage.

## ROI Align and ROI heads

ROI Align samples a fixed-size feature representation for each proposal from the appropriate pyramid level. The box head turns these features into foreground/background class logits and class-specific box refinements. Postprocessing applies score filtering, clipping, non-maximum suppression, and a per-image cap.

Its two training losses are:

| Key | Responsibility |
|---|---|
| `loss_classifier` | classify ROI samples as background or an object class |
| `loss_box_reg` | refine positive ROI samples toward final target boxes |

The exact Faster R-CNN loss set is therefore `loss_classifier`, `loss_box_reg`, `loss_objectness`, and `loss_rpn_box_reg`. The project checks that each is a finite scalar, sums all returned losses into `loss_total`, then backpropagates. Numeric values depend on data, initialization, model, and device; no fixed value is expected.

## Mode-dependent public API

| Mode | Call | Return |
|---|---|---|
| training | `model.train(); model(images, targets)` | mapping of the four scalar losses |
| evaluation | `model.eval(); model(images)` under `torch.inference_mode()` | one prediction mapping per image |

Each prediction has `boxes: float32 [M,4]`, `labels: int64 [M]`, and `scores: float32 [M]`. `M` varies per image. Evaluation does not accept targets to request losses; training does not return predictions for metric interpretation.

Run the real contract offline:

```bash
uv run python examples/03_model_contract.py
```

Expected output names the four losses and the three prediction keys. The example constructs `fasterrcnn_mobilenet_v3_large_320_fpn` with `weights="none"` and synthetic tensors. It performs forward passes only, does not update parameters, and provides no accuracy evidence.

## Failure and evidence boundaries

Labels must start at 1 because 0 is background. Boxes must be finite, positive-area, zero-based continuous `xyxy`. Images and targets must remain one-to-one lists. A dry run extends the example by performing one optimizer update on prepared data; a bounded training run adds validation and artifacts. Neither is the [separately recorded full-VOC result](../recorded-run/README.md).

Continue with [Tutorial 04](../tutorial/04-training.md) for optimization and artifact ownership, or read the [model reference](../reference/model-zoo.md) to compare the two Faster R-CNN backbones with SSDLite.
