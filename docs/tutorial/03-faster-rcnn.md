# Tutorial 03: How Faster R-CNN Finds Objects

[简体中文](03-faster-rcnn.zh-CN.md) | [Tutorial index](README.md)

Faster R-CNN is a two-stage detector. It first finds regions that may contain
objects, then decides what each region contains and refines its box. These two
steps explain the four losses in the training log.

## Why the input is a list

Images in one batch can have different sizes, so the model receives:

```text
images  = [Tensor[3, H1, W1], Tensor[3, H2, W2], ...]
targets = [{boxes, labels, ...}, {boxes, labels, ...}, ...]
```

Torchvision resizes, normalizes, and pads images inside the model, then maps
output boxes back to the original image sizes. The caller does not stretch
every image to one fixed shape first.

## Stage one: the RPN proposes regions

The backbone and FPN convert an image into feature maps at several scales. The
Region Proposal Network (RPN) searches those features for regions that may
contain objects. It only asks whether a region looks like an object; it does not
yet distinguish person, dog, or another VOC class.

The RPN produces two training losses:

- `loss_objectness`: object versus background for proposed regions.
- `loss_rpn_box_reg`: how proposal boxes should move and resize.

## Stage two: the ROI head classifies and refines

ROI Align extracts a fixed-size feature for each proposed region. The ROI head
chooses a class and refines the box. It produces two more losses:

- `loss_classifier`: background and object-class classification.
- `loss_box_reg`: final bounding-box adjustment.

The project adds the four losses into `loss_total` for backpropagation and
logging. Values vary from batch to batch; one smaller loss by itself does not
prove that a model is better.

## Training and prediction return different values

| State | Call | Return value |
|---|---|---|
| Training | `model(images, targets)` | Four losses |
| Evaluation/prediction | `model(images)` | `boxes`, `labels`, and `scores` for each image |

You can inspect this behavior with a randomly initialized model:

```bash
uv run python examples/03_model_contract.py
```

The example only performs forward passes. It does not train or produce a model
score. It prints training loss names, then prediction fields.

The small variable-size batch example is:

```bash
uv run python examples/02_detection_batch.py
```

## What happens in one parameter update

The core training order is:

```text
clear old gradients
-> compute four losses
-> sum them into loss_total
-> backpropagate
-> update parameters with the optimizer
```

The Kaggle runner performs one-batch dry run to confirm that this path works
before beginning 26 epochs. Losses in the log come from the torchvision model;
they are not scores invented by this project.

## Common points of confusion

- Object labels start at `1`; `0` is reserved for background.
- The RPN proposes regions but does not predict final VOC classes.
- Training needs targets; prediction does not.
- Finite losses from random weights do not mean the model has learned.
- A prediction `score` is not IoU or mAP.

Continue to [training](04-training.md) to connect these losses to the Kaggle log
and `metrics.csv`.
