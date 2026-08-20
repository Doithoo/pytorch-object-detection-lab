# From Bounding Boxes to Kaggle Training

[简体中文](learning-path.zh-CN.md) | [Tutorial index](README.md)

This route is for readers who know basic Python and have seen tensors and
gradients, but have not trained an object detector end to end. You do not need
a local GPU. The Kaggle runner performs the complete sequence:

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

## 1. Know what the model must produce

Object detection answers both "what is in the image?" and "where is it?" A
prediction normally contains:

- `boxes`: xyxy bounding boxes with shape `[N, 4]`.
- `labels`: one class index for each box.
- `scores`: the model confidence for each prediction.

After [detection basics](00-basics.md), you should be able to read one target.
There is no need to memorize many formulas before continuing.

## 2. Meet the training data

The project uses Pascal VOC 2007: 2,501 training images, 2,510 validation
images, and 4,952 test images across 20 object classes. The Kaggle runner
downloads the official archives and creates the manifests used for training.

The [VOC data chapter](02-data-and-boxes.md) explains why XML coordinates are
converted and why `difficult` objects are handled differently during
evaluation. On a first read, focus on the meaning rather than manifest hashes
or internal file-writing details.

## 3. Understand the two Faster R-CNN outputs

During training, the model receives images and targets and returns four losses:
classification, box regression, RPN classification, and RPN box regression.
During evaluation, it receives images and returns boxes, labels, and scores.

The [Faster R-CNN chapter](03-faster-rcnn.md) connects the RPN and ROI head.
Start with the two ideas of proposing regions and then classifying and refining
them.

## 4. Submit training to Kaggle

Follow the [Kaggle guide](../guides/kaggle.md) to:

1. Install the Kaggle CLI and sign in.
2. Replace the account name in the kernel metadata.
3. Submit the runner and confirm a T4-or-newer GPU and Internet on the web page.
4. Wait for `COMPLETE`, then download only `artifacts/.*`.

Training takes about 50-60 minutes. If the log continues to print heartbeats
and epoch updates, there is no need to change the configuration or resubmit.

## 5. Read the training history

Open `metrics.csv` and start with:

- `epoch`: the completed epoch.
- `loss_total`: the sum of training losses.
- `valid_map_50_95`: the main validation selection metric.
- `valid_map_50`: validation performance at IoU 0.5.

Loss and mAP measure different things and do not need to move together. The
project selects `best.pt` with validation `map_50_95`, not simply the last
epoch. The saved run selected epoch 18 and continued training through epoch 26.

## 6. Inspect a real evaluation

Look under `evaluation/`:

- `evaluation.json` contains the test summary.
- `per_class.csv` shows differences across all 20 classes.
- `errors.csv` records false positives, misses, and localization errors.
- `visualizations/` shows where the model succeeded or failed.

Practice with the [saved result](../recorded-run/README.md), then inspect your
own files. The published Kaggle result is
`mAP@0.5:0.95 = 0.322312` and `mAP@0.5 = 0.609917`.

## 7. Choose what to explore next

After the first run:

- To understand training code, read [training](04-training.md) and the
  [code tour](../concepts/code-tour.md).
- To analyze the model, read [evaluation and prediction](05-evaluation-and-inference.md).
- To change models, read [choosing a model](../guides/using-models.md).
- To use your own data, read the [custom data guide](../guides/using-your-data.md).
- If you have a local GPU, use the local command at the end of the training chapter.

Change one important setting at a time and keep the original `config.yaml` and
`metrics.csv`. This makes the result easier to understand than changing many
parameters together.
