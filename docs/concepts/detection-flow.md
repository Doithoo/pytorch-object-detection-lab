# End-to-End Detection Flow

[Simplified Chinese](detection-flow.zh-CN.md) | [Code tour](code-tour.md)

This page is for readers connecting command-line values, data tensors, torchvision modes, and published artifacts. It describes ownership across the actual training, evaluation, and prediction paths.

## Command to preflight

```text
detect train
  -> strict defaults < YAML < repeated --set
  -> final runtime --device override
  -> load dataset.yaml
  -> preflight manifests, class count, device, output, weight cache
  -> resolve auto device and construct model
```

Configuration errors stop before data/model work. Metadata loading proves `dataset.yaml` can be parsed; preflight then requires all three CSV files and metadata, checks `expected_num_classes == len(class_names)+1`, rejects unavailable explicit accelerators and unwritable output ancestry, and emits a notice for an uncached named backbone weight. A notice means construction may need network access, not that download has already happened.

## Source to immutable manifest identity

```text
VOC split IDs + JPEG bytes + XML bytes
  -> validate disjoint membership, pairing, decoding, dimensions, classes, boxes
  -> SHA-256 per split
  -> identity(name, classes, coordinate convention, split hashes)
  -> train.csv / valid.csv / test.csv + dataset.yaml + provenance
```

Preparation stages all outputs and atomically replaces the manifest directory. CSV rows reference source files; source data is not copied into the manifest. Runtime therefore needs both the manifest directory and matching source root. A later source change requires a new preparation and identity.

## Manifest row to variable batch

One row is decoded into `image: float32 [3,H,W]` RGB in `[0,1]` and a target mapping whose object fields share `N`. VOC boxes become zero-based continuous `float32 [N,4]`; labels are `int64 [N]` with background reserved as 0. `image_id`, `area`, `iscrowd`, and `difficult` carry identity and evaluation semantics.

Training removes difficult objects before random horizontal flip. Evaluation/inspection retains them. Degenerate-box filtering applies the same keep mask to all aligned fields. Empty targets remain shaped, valid tensors.

```text
sample 1: Tensor[3,H1,W1] + target[N1]
sample 2: Tensor[3,H2,W2] + target[N2]
  -> detection_collate
list[Tensor] + list[target]
```

The project does not resize, pad, or stack this batch. The torchvision detector's transform owns normalization, aspect-aware resize, padding, and mapping final boxes back to caller coordinates.

## One module, two API modes

```text
model.train(); model(images, targets)
  -> mapping of named scalar losses
  -> finite checks -> sum -> backward -> optional clip -> optimizer step

model.eval(); model(images) under inference_mode
  -> list[{boxes, labels, scores}]
  -> metric / JSON / error analysis / visualization
```

For Faster R-CNN the loss keys are `loss_classifier`, `loss_box_reg`, `loss_objectness`, and `loss_rpn_box_reg`; `loss_total` is their project-side sum. Other registered families can return different named losses. Passing the wrong arguments or interpreting a result under the wrong mode violates the model API.

Run both modes with random synthetic tensors:

```bash
uv run python examples/03_model_contract.py
```

Expected output lists Faster R-CNN training loss keys and eval keys `boxes`, `labels`, `scores`. It checks construction and input/output shapes only. It downloads no weights, trains nothing, and publishes no metric.

## Training, validation, and atomic run artifacts

The trainer averages each loss by image count. After each epoch, validation sends all detections returned by the model to AP/AR without a project score filter. The orchestrator appends one history row, steps the optional fixed StepLR, and updates `best.pt` only when validation `map_50_95` strictly exceeds its previous best. `last.pt` always records the completed epoch.

```text
artifacts/<run>/
  config.yaml   resolved configuration
  run.yaml      environment + manifest identity
  metrics.csv   epoch losses + validation AP/AR/counts
  best.pt       best validation checkpoint
  last.pt       latest resumable checkpoint
```

Text files and checkpoints are written through temporary files and `os.replace`. A fresh run rejects an existing run directory. Dry run stops after one integrated optimizer update and writes none of these normal artifacts.

## Checkpoint to evaluation or prediction

Both consumers load schema v1 through `torch.load(..., weights_only=True)`, validate exact preprocessing, rebuild the registered model with `weights="none"`, and load saved state.

Evaluation additionally loads the saved resolved config, requires the current manifest identity to match, reads a labeled split, sends model output to metrics, applies separate serialization and error thresholds, and atomically publishes JSON, CSV, and ranked PNG images. Prediction needs only checkpoint plus new images. It restores ordered classes and manifest provenance, but does not load VOC data or claim AP.

Directory prediction recursively processes `.jpg`, `.jpeg`, and `.png`, records unreadable images, and publishes a complete staged output tree. Single-image prediction writes one JSON and PNG with overwrite protection; its JSON is atomic, while its PNG is saved directly.

## What each check tells you

`show-config` checks value resolution. `inspect-data` checks target loading. `train --dry-run` checks one update. A small run checks output creation. Validation evaluation checks metrics on the configured subset. None establishes a complete VOC result, and the reference YAML is not a completed run.

Read [how Faster R-CNN works](how-faster-rcnn-works.md) for internal detector responsibilities, [checkpoint schema](../reference/checkpoint-schema.md) for restoration, and [metrics](../reference/metrics.md) for output semantics.
