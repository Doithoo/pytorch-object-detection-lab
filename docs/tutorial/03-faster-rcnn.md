# Tutorial 03: Faster R-CNN as a Tensor Pipeline

[Simplified Chinese](03-faster-rcnn.zh-CN.md) | [Tutorial index](README.md)

Prerequisites are the list-based batch and xyxy target contract from Tutorials
00 and 02. This chapter uses random synthetic input and `weights=none`, so it
does not need VOC data, a checkpoint, or network access. Its goal is to explain
responsibilities and mode-dependent APIs, not to reproduce torchvision internals.

## The detector owns padding and resize

The caller passes `list[Tensor[3, Hi, Wi]]`. Torchvision's generalized detector
transform normalizes and resizes each image, then pads them into an internal
image-list tensor `[B, 3, Hpad, Wpad]` while retaining each resized image size.
That metadata is needed to map final boxes back to the caller's coordinates.

The target list remains aligned one-for-one with the images. During training,
each target supplies foreground boxes `[Ni, 4]` and labels `[Ni]`; during
evaluation, targets are not passed.

## Backbone and FPN: pixels become feature maps

The backbone turns the padded image tensor into spatial feature maps. A Feature
Pyramid Network (FPN) combines deep semantic information with finer spatial
resolution and exposes several levels, conceptually:

```text
image list [B, 3, Hpad, Wpad]
    -> backbone + FPN
feature level k [B, C, Hk, Wk], for several scales k
```

Small objects can use a finer level; larger objects can use a coarser one. These
features are not boxes and do not yet carry final VOC class decisions.

## RPN: feature maps become class-agnostic proposals

The Region Proposal Network (RPN) evaluates anchors across FPN levels. For each
anchor it predicts objectness and a box adjustment. After decoding, clipping,
ranking, and suppression, the next stage receives a variable number of proposal
boxes per image, each shaped `[Ki, 4]`.

The RPN contributes two training values:

- `loss_objectness`: whether sampled anchors contain an object rather than
  background.
- `loss_rpn_box_reg`: how well positive anchors are adjusted toward targets.

RPN proposals are class-agnostic. Calling them `person` or `dog` at this stage
mixes responsibilities between the proposal and ROI heads.

## ROI heads: proposals become class and box predictions

ROI Align samples a fixed spatial feature for each proposal from the appropriate
FPN level. The box head converts those features into class logits, including a
background class, and class-specific box adjustments. Per-image postprocessing
then returns a variable number of detections:

```text
prediction["boxes"]   float32 [M, 4]
prediction["labels"]  int64   [M]
prediction["scores"]  float32 [M]
```

The ROI heads contribute the other two training values:

- `loss_classifier`: foreground/background and object-class classification.
- `loss_box_reg`: final box refinement for positive ROI samples.

For Faster R-CNN, the exact four torchvision loss keys are therefore
`loss_classifier`, `loss_box_reg`, `loss_objectness`, and `loss_rpn_box_reg`.
The project sums them into `loss_total` for backpropagation and logging. Their
numeric values depend on initialization, inputs, and device; this tutorial makes
no expected-loss claim.

## Train mode and eval mode are different APIs

Torchvision detection models change both accepted inputs and returned values:

| Mode | Call | Return |
|---|---|---|
| training | `model(images, targets)` | dictionary of scalar loss tensors |
| evaluation | `model(images)` under inference mode | one prediction dictionary per image |

Run the real maintained model contract:

```bash
uv run python examples/03_model_contract.py
```

Expected output lists the four training keys above and evaluation keys
`boxes`, `labels`, and `scores`. The command constructs
`fasterrcnn_mobilenet_v3_large_320_fpn` with random weights. It performs forward
passes only, learns nothing, and publishes no score.

`examples/02_detection_batch.py` shows the exact list container delivered to
both modes:

```bash
uv run python examples/02_detection_batch.py
```

Expected shapes remain `(3, 16, 20)` and `(3, 12, 24)` before the model-owned
transform. The model may resize and pad them internally; that does not change the
dataset's coordinate convention.

## From four losses to one update

One production optimization step is:

```text
model.train()
optimizer.zero_grad(set_to_none=True)
losses = model(images, targets)
loss_total = sum(losses.values())
loss_total.backward()
optimizer.step()
```

Chapter 04 runs this path against prepared data. The tiny
`examples/04_minimal_training_loop.py` isolates the mechanics with a fake
two-loss detector; do not mistake its loss dictionary for Faster R-CNN's exact
four-loss contract.

## Common failure boundaries

- Calling a train-mode detector without targets: the training API is incomplete.
- Passing targets in eval mode and expecting losses: eval mode returns
  predictions instead.
- Passing one stacked tensor instead of a list: the public detection contract is
  violated and original sizes become ambiguous.
- Using label `0` for an object: it is interpreted as background.
- Treating RPN objectness as VOC classification: the RPN is class-agnostic.
- Interpreting finite random-weight losses or prediction keys as learned quality:
  only the software contract was exercised.

Continue to [Tutorial 04](04-training.md) to perform one update, then distinguish
a dry run, a bounded learning run, and a full experiment.
