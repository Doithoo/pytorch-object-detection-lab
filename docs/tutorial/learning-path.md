# Object Detection Learning Path

[Simplified Chinese](learning-path.zh-CN.md) | [Tutorial index](README.md)

This route is for a learner who knows basic tensors and gradients but has not yet
completed a two-stage object-detection workflow. Move through
`download -> prepare -> inspect -> dry run -> train -> evaluate -> predict` and
state what each stage proves before continuing.

## 0. Establish the locked environment

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect show-config --config configs/learning_minimal.yaml
```

Expected version output is `0.1.0`. The resolved config shows random weights,
two epochs, sample limits of 32/16/16, and automatic device selection. If these
commands fail, use [Tutorial 01](01-environment.md) before touching data or GPU
settings.

Completion check: explain why `weights: none` is an offline model-construction
policy but not a claim that VOC is installed.

## 1. Identify detection tensors and lists

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
```

Predict the areas `192` and `180` before reading example 01 output. In example
02, explain why two images with shapes `(3,16,20)` and `(3,12,24)` remain a list,
why targets form a parallel list, why boxes are `[N,4]`, and why object labels
start at 1. Read [Tutorial 00](00-basics.md) if any answer is unclear.

Completion check: write a valid empty target shape and compute the IoU of two
partially overlapping boxes by hand.

## 2. Build a trusted prepared-data boundary

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split valid --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset_preview.png
```

The first command is the network boundary and verifies the two official VOC MD5
checksums. Preparation validates and atomically publishes fixed manifests plus a
content-derived identity. Inspection prints structured counts/ranges. Preview
renders whichever ordinary and difficult annotations occur in the first selected
rows; difficult boxes appear only when those rows contain them. Treat the
manifest identity as immutable for all runs you compare. Read
[Tutorial 02](02-data-and-boxes.md).

Completion check: explain `(xmin-1, ymin-1, xmax, ymax)`, identify the labeled
difficult target in the explicitly synthetic
[target anatomy diagram](../assets/detection-target-anatomy.png), and distinguish
inspected-image counts from full-split counts.

## 3. Cross the real model-mode boundary

```bash
uv run python examples/03_model_contract.py
```

Expected train-mode keys are `loss_classifier`, `loss_box_reg`,
`loss_objectness`, and `loss_rpn_box_reg`; eval-mode keys are `boxes`, `labels`,
and `scores`. The example uses random weights and synthetic input. It proves the
torchvision contract but performs no learning. Follow the tensor responsibilities
in [Tutorial 03](03-faster-rcnn.md).

Completion check: trace image list -> padded image list -> backbone/FPN -> RPN
proposals -> ROI predictions, and say which two losses belong to each head.

## 4. Perform one optimization step

```bash
uv run python examples/04_minimal_training_loop.py --lr 0.1
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

The tiny example makes one parameter change visible. The dry run then updates the
configured detector on one prepared-data batch and ends with `dry-run OK`. It
does not write a checkpoint or report quality. Use [Tutorial 04](04-training.md)
to distinguish this probe from training evidence.

Completion check: point to the operations that clear old gradients, construct a
scalar loss, compute new gradients, and modify parameters.

## 5. Complete a bounded learning run

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --device cpu
```

Inspect `config.yaml`, `run.yaml`, `metrics.csv`, `best.pt`, and `last.pt` under
`artifacts/first-detector`. The 32/16/16 sample limits and two epochs make this a
workflow-learning run, not a complete VOC benchmark. Explain why validation
`map_50_95` selects `best.pt` and why it may differ from `last.pt`.

Completion check: recover the exact manifest identity, weight policy, device,
sample limits, selected epoch, and four loss columns from artifacts rather than
from memory.

## 6. Evaluate evidence, then predict

```bash
uv run detect evaluate --checkpoint artifacts/first-detector/best.pt --split valid --output-dir artifacts/first-detector/evaluation-valid --device cpu
uv run detect predict --checkpoint artifacts/first-detector/best.pt --image docs/assets/detection-target-anatomy.png --output-dir artifacts/prediction --device cpu
```

The evaluate command requires matching prepared data and writes AP/AR,
predictions, per-class rows, categorized errors, and ranked evidence images. The
prediction command needs only the checkpoint and the shipped synthetic teaching
diagram. It writes `detection-target-anatomy.json` and
`detection-target-anatomy.png` under `artifacts/prediction`; this checks inference
and artifact mechanics, not detector quality. Read
[Tutorial 05](05-evaluation-and-inference.md), then explain one missed or false
positive case with both its CSV row and visualization.

Completion check: say why IoU and score thresholds are separate, why difficult
matches are ignored, and why validation rather than test is used while making
choices.

## 7. Compare one controlled change

Create two distinctly named bounded runs that share the same manifest identity,
seed, limits, and evaluation protocol, changing one intended configuration field.
Then run:

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

The report selects the best row per run, verifies equal manifest identity, and
shows semantic configuration differences. It intentionally excludes the
operational fields `run_name`, `output_dir`, `device`, and `data.num_workers`.
Pair that table with curves and visual errors. Two bounded runs establish only
what happened in those runs.

## Evidence boundary

Synthetic examples prove local tensor and API contracts. A dry run proves one
integrated update. A bounded run proves the artifact and evaluation path for its
configured subset. None is a complete Pascal VOC benchmark. The separate
[recorded full-VOC run](../recorded-run/README.md) preserves the provenance,
scope, metrics, runtime, checkpoint hash, and real images needed for that claim.
