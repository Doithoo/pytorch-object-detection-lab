# Tutorial

[Simplified Chinese](README.zh-CN.md) | [Documentation index](../README.md)

Use this route for a first end-to-end pass. Read the chapters in order because each one introduces a contract needed by the next. The operational sequence is exactly `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`.

## Before you start

Install Python 3.10-3.12 and uv, clone the repository, and run `uv sync --locked --extra dev`. Chapters 00, 01, and 03 use synthetic inputs and do not need VOC files. Follow the repository's main [data preparation workflow](../../README.md) to download, validate, and create manifests before Chapter 02; that chapter then explains VOC coordinates and difficult objects and previews the prepared split. The default learning configuration uses `weights: none`, so its model path does not download weights.

## Chapter map

| Chapter | Read it when | Prerequisite | Expected output or decision |
|---|---|---|---|
| [Learning path](learning-path.md) | You need the whole workflow before details | Installed environment | CLI version output and a map of the seven stages |
| [00 - Detection basics](00-basics.md) | Boxes, labels, empty targets, or variable image sizes are new | Basic Python and tensor indexing | Printed xyxy boxes, integer labels, areas, and variable batch shapes |
| [01 - Environment](01-environment.md) | You need to verify the locked environment and offline weight policy | Repository dependencies installed | Resolved learning YAML with `weights: none` and bounded sample limits |
| [02 - VOC data and boxes](02-data-and-boxes.md) | Prepared data is available and you need to understand its coordinates and difficult objects | Prepared manifests and matching source images from the main data preparation workflow | Coordinate-conversion notes and `artifacts/dataset_preview.png` with ordinary and difficult boxes |
| [03 - Faster R-CNN](03-faster-rcnn.md) | You need to understand training losses versus evaluation predictions | Chapters 00-02 concepts | Named detector losses from a synthetic example; no learned checkpoint |
| [04 - Training](04-training.md) | Prepared data and model contracts are clear | Local manifests and source images | One-update dry-run diagnostics, then a bounded run directory with config, provenance, metrics, and checkpoints |
| [05 - Evaluation and inference](05-evaluation-and-inference.md) | You have a checkpoint and want reports or predictions | Matching prepared data for evaluation; only local images for prediction | Evaluation JSON/CSV/evidence files or prediction JSON/PNG files |

## Checkpoints along the route

Before training, inspect both a machine-readable summary and rendered samples:

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --limit 4
```

The first command prints counts, class frequencies, box ranges, and difficult-object information. The second writes `artifacts/dataset_preview.png`. Neither command proves that optimization works.

Then cross the training boundary with one update:

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

Expected output includes image shapes, target counts, named finite losses, and `dry-run OK`. A dry run writes no checkpoint. A normal run with the same config is bounded to 32 training, 16 validation, and 16 test samples and writes learning artifacts; it is not a full VOC benchmark.

## What completion means

Finishing the tutorial means you can trace data and artifacts through the whole workflow. Synthetic examples establish local contracts. A bounded learning run establishes that the integrated path executes and can update parameters. The [recorded full-VOC run](../recorded-run/README.md) shows the higher evidence level: exact provenance, validation selection, all-split scope, test metrics, checkpoint hash, runtime, and real failure images.
