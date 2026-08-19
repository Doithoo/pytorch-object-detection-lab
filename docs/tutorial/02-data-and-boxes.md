# Tutorial 02: Meet VOC Data and Bounding Boxes

[简体中文](02-data-and-boxes.zh-CN.md) | [Tutorial index](README.md)

Pascal VOC 2007 is a good first object-detection dataset: it is manageable,
well documented, and contains 20 familiar object classes. This project uses
the official splits:

| Split | Images | Purpose |
|---|---:|---|
| train | 2,501 | Update model parameters |
| valid | 2,510 | Compare epochs and select `best.pt` |
| test | 4,952 | Final evaluation after training |

The Kaggle runner downloads, extracts, and prepares these files automatically.
You do not upload a dataset manually for the first run.

## What one target contains

VOC provides one XML file per image. The project converts it to the target
dictionary expected by torchvision. The most important fields are:

```text
boxes:      FloatTensor[N, 4]
labels:     Int64Tensor[N]
image_id:   Int64Tensor[1]
area:       FloatTensor[N]
iscrowd:    Int64Tensor[N]
difficult:  BoolTensor[N]
```

`N` is the number of objects in the image. When there are no ordinary objects,
`boxes` still has shape `[0, 4]`, not a one-dimensional empty tensor.

## How VOC coordinates are converted

VOC XML coordinates are one-based and inclusive. The project converts them
once:

```text
(xmin - 1, ymin - 1, xmax, ymax)
```

For example, `(11, 21, 50, 70)` becomes `[10, 20, 50, 70]`. The converted
width is `40`, height is `50`, and area is `2000`. Later transforms and model
code must not subtract or add one again.

![Teaching diagram of an image, bounding boxes, and target fields](../assets/detection-target-anatomy.png)

This is a teaching diagram, not a training result. Green boxes are ordinary
objects and the dashed orange box is a difficult object.

## Difficult objects

VOC uses `difficult=1` for objects that cannot be identified or localized
reliably. This project:

- Excludes difficult objects from training losses.
- Keeps them in validation and test targets.
- Does not count a prediction that only matches a difficult object as an
  ordinary false positive.

This avoids training on ambiguous targets while preserving their information
for evaluation and visualization.

## Data preparation on Kaggle

The runner:

1. Downloads the official train/validation and test archives.
2. Checks the official MD5 values.
3. Extracts them under `/kaggle/working/data`.
4. Creates `train.csv`, `valid.csv`, `test.csv`, and `dataset.yaml`.
5. Reads one batch before training to confirm that images and boxes reach the
   model.

These log lines show that the download completed:

```text
{"phase": "download_voc2007", "status": "started"}
{"phase": "download_voc2007", "status": "completed", ...}
```

The prepared-data identity is written to `run.yaml` and checkpoints so an
evaluation can confirm that it uses the same data. For a first pass, think of
it as the version number of the prepared dataset.

## Optional: inspect VOC locally

To open the images and annotations yourself, run from the repository root:

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset_preview.png
```

`inspect-data` prints image sizes, object counts, classes, and box ranges. The
preview draws annotations on real images so you can check labels and box
placement. `--limit` controls how many images you inspect; it does not change
the official split.

## What to check when something looks wrong

- Every box is shifted by one pixel: check for a repeated `xmin - 1` or
  `ymin - 1` conversion.
- `boxes` has the wrong shape: an empty target must still be `[0, 4]`.
- A training image has no targets: check whether all its objects are difficult.
- Kaggle cannot download data: confirm Internet is enabled, then check whether
  the official host is temporarily unavailable.
- Counts are not `2501 / 2510 / 4952`: confirm you have complete official
  VOC 2007 data.

Continue to [Faster R-CNN](03-faster-rcnn.md) to see how images and targets
enter the model.
