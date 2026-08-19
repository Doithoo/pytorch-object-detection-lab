# Tutorial 05: Read Evaluation Results and Predictions

[简体中文](05-evaluation-and-inference.zh-CN.md) | [Tutorial index](README.md)

You can learn this chapter from the saved Kaggle v7 result without first
downloading the 145 MB checkpoint. Start in
[`../recorded-run/evaluation`](../recorded-run/evaluation).

## Start with the summary

[`evaluation.json`](../recorded-run/evaluation/evaluation.json) records the
evaluation of all 4,952 VOC 2007 test images:

| Metric | Kaggle v7 result |
|---|---:|
| `map_50_95` | **0.322312** |
| `map_50` | **0.609917** |
| `map_75` | 0.302681 |
| `mar_100` | 0.415008 |

These are unitless values from 0 to 1, not percentages. `map_50_95` requires
both correct classes and increasingly accurate boxes, so it is normally lower
than `map_50`, which uses only IoU 0.5.

## What mAP measures

A prediction must have the same class as a target and reach the selected IoU to
match it. IoU is the intersection area of two boxes divided by their union.

- `map_50`: AP at IoU 0.5.
- `map_75`: AP at the stricter IoU 0.75.
- `map_50_95`: AP averaged over 0.50, 0.55, ..., 0.95.
- `mar_100`: average recall with at most 100 predictions per image.

AP combines precision and recall across confidence thresholds, so one image or
one score threshold cannot replace it.

## Inspect differences between classes

Open [`per_class.csv`](../recorded-run/evaluation/per_class.csv). It lists
`map_50_95` and `mar_100` for each of the 20 VOC classes. Differences can come
from object size, occlusion, appearance variation, data volume, and class
confusion.

Find a few strong and weak classes, then look for concrete causes in
`errors.csv` and the images. A class AP alone does not show that the model
"understands" or "does not understand" an object.

## Move from error rows back to images

[`errors.csv`](../recorded-run/evaluation/errors.csv) contains four record types:

- `missed`: an ordinary target had no qualifying prediction.
- `false_positive`: a prediction did not match an ordinary target of its class.
- `localization`: the class may be right, but overlap did not reach the threshold.
- `ignored`: the prediction only matched a difficult object and is not counted
  as an ordinary false positive.

Start with the real summary image:

![Targets and predictions for test image 000001 from the Kaggle model](../recorded-run/evaluation/visualizations/summary.png)

Green boxes are ordinary targets, dashed orange boxes are difficult targets,
and blue boxes are predictions. Then inspect:

- [A false-positive example](../recorded-run/evaluation/visualizations/false_positive-01-009040.png)
- [A missed-object example](../recorded-run/evaluation/visualizations/missed-01-006500.png)

One image cannot represent the whole test set, but it can suggest the next
question: small object, occlusion, class confusion, poor localization, or a
duplicate prediction? Return to the CSV for its class, score, and IoU.

## Why difficult objects are handled separately

VOC difficult targets are hard to identify or localize reliably. They are not
counted as ordinary targets and are not marked as missed. A prediction that only
matches a difficult target is `ignored`, not `false_positive`.

This is why the data path must retain `difficult` / `iscrowd` information.

## Validation and test have different jobs

Use validation during training to:

- Compare epochs.
- Select `best.pt`.
- Adjust models and hyperparameters.

Use test once after those choices are fixed. The published run reached its best
validation metric at epoch 18 and evaluated test only after all 26 epochs were
complete. Repeatedly changing the model after looking at test removes that
independence.

## Predict with your downloaded checkpoint

After downloading `best.pt` from Kaggle, predict one local image:

```bash
uv run detect predict --checkpoint kaggle-output/reference-fasterrcnn/best.pt --image image.jpg --output-dir artifacts/prediction --device cpu --score-threshold 0.5
```

The output directory contains JSON and PNG files with the same stem. JSON keeps
floating-point boxes, classes, and scores; PNG is convenient to inspect.
Raising `--score-threshold` hides more low-confidence predictions, but it does
not retrain the model or improve box locations.

Predict a directory with:

```bash
uv run detect predict --checkpoint kaggle-output/reference-fasterrcnn/best.pt --input-dir images --output-dir artifacts/predictions --device cpu --score-threshold 0.5
```

Prediction needs only the checkpoint and images, not VOC data. CPU inference
works, although it is slower than GPU inference.

## Re-evaluate your Kaggle result

The Kaggle runner already evaluates test. You only need to run evaluation again
when changing visualization thresholds or regenerating
files in an environment with matching VOC data. See the
[metrics reference](../reference/metrics.md) for all options.

You have now followed the main route from boxes through Kaggle training to
error analysis. Continue with [comparing configurations](../guides/experiments.md),
[changing models](../guides/using-models.md), or
[using your own data](../guides/using-your-data.md).
