# Use Your Own VOC-Shaped Data

[简体中文](using-your-data.zh-CN.md) | [Dataset format](../reference/dataset-format.md)

If your data already follows a Pascal VOC-shaped JPEG, XML, and split-text layout, you can reuse the project's preparation, training, and evaluation code. The provider infers the foreground classes from all validated XML files and writes a stable sorted label mapping into `dataset.yaml`. Official VOC 2007 keeps its published 20-class order.

## Directory layout

```text
my-data/
└── VOCdevkit/
    └── VOC2007/
        ├── Annotations/
        ├── ImageSets/Main/
        │   ├── train.txt
        │   ├── val.txt
        │   └── test.txt
        └── JPEGImages/
```

Every image ID needs matching `.jpg` and `.xml` files. XML class names must be non-empty
and consistent across the source; `difficult` may be absent or `0` / `1`, and
coordinates use VOC one-based inclusive format.

## Prepare and inspect data

Explicitly allow counts that differ from official VOC:

```bash
uv run detect prepare-data --data-dir my-data --manifest-dir data/my-manifests --allow-nonstandard-counts
uv run detect inspect-data --manifest-dir data/my-manifests --data-dir my-data --split train --limit 16
uv run python scripts/preview_dataset.py data/my-manifests --data-dir my-data --split train --limit 4 --output artifacts/my-data-preview.png
```

Open the preview and check labels and box placement. Fix count, class,
coordinate, or image-size problems in source XML, then regenerate manifests.

## Check training with a few samples

Override the data paths and update the expected class count before training:

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set data.data_dir my-data --set data.manifest_dir data/my-manifests --set run_name my-data-check
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir my-data --set data.manifest_dir data/my-manifests --set model.expected_num_classes 4 --set run_name my-data-check --dry-run --device cpu
```

`dry-run OK` means one image batch and its targets completed an update. It does
not save a model.

## Begin a training run

Copy and adjust a configuration for your data size, choose a new `run_name`,
then train on Kaggle or a compatible local GPU. Metrics from non-official data
describe only your split and are not directly comparable with the project's
VOC 2007 values.

Keep `config.yaml`, `run.yaml`, `metrics.csv`, `best.pt`, and `last.pt`. During
evaluation, use the same `data/my-manifests` that produced the checkpoint.
