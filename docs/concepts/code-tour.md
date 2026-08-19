# Code Tour: From Command to Result

[Simplified Chinese](code-tour.zh-CN.md) | [Detection flow](detection-flow.md)

This page is for contributors who want to locate ownership before changing code. Follow one command through the package instead of reading modules alphabetically.

```text
detect train
  -> object_detector.cli._train
  -> object_detector.config.load_config
  -> object_detector.training.train.run_training
  -> manifest metadata + preflight
  -> VocDetectionDataset + list collate
  -> model registry + torchvision constructor
  -> trainer.train_one_epoch / DetectionMetric
  -> atomic config, metadata, metrics, checkpoints
```

## Commands and configuration

`src/object_detector/cli.py` owns argparse names, runtime-only options, concise stdout, and conversion of handled `ValueError`, `RuntimeError`, or `OSError` into `error: ...` with exit code 2. It delegates domain work. `src/object_detector/config.py` owns dataclass defaults, strict YAML merge, typed `--set` values, source tracking, path construction, and value validation. `src/object_detector/preflight.py` owns manifest-file/class-count/device/output/cache readiness and `auto` device resolution.

Run `uv run detect show-config --config configs/learning_minimal.yaml` to inspect only this first part. Expected output is resolved YAML plus `sources`; no data, model, network, or output work occurs.

## Data loading

Read these in source order:

1. `data/schema.py` defines VOC classes and parsed annotation records.
2. `data/voc.py` parses XML and converts one-based inclusive boxes.
3. `data/manifest.py` validates every split sample, hashes source content, and atomically publishes manifests.
4. `data/dataset.py` resolves rows into float RGB tensors and torchvision targets, with difficult filtering only for training.
5. `data/transforms.py` keeps boxes and object fields aligned through geometry changes.
6. `data/inspection.py` computes bounded summaries and renders target previews.

`detection_collate` deliberately returns two lists. It does not pad or stack variable-sized images. When labels, difficult flags, dimensions, or geometry look wrong, debug the data path before the model.

## Models

`models/spec.py` defines registry metadata and constructor type. `models/registry.py` owns stable names, supported weight policies, cache-path derivation, reserved parameters, and close-name errors. `models/torchvision_models.py` is the only layer that translates project policies into torchvision constructor arguments. Backbone, FPN, RPN, ROI Align, heads, model-owned transforms, and NMS remain torchvision responsibilities.

`uv run detect list-models` and `uv run detect model-info fasterrcnn_resnet50_fpn` inspect metadata without constructing a model. `examples/03_model_contract.py` constructs a model and runs both modes with synthetic input, but does not measure learned quality.

## Training and checkpoints

`training/trainer.py` owns batch movement, loss summation, finite-scalar checks, backward, optional gradient clipping, optimizer stepping, one epoch, and dry-run diagnostics. `training/train.py` owns seeding, datasets/loaders, optimizer/scheduler construction, validation, best/last selection, resume semantic checks, history, and run artifacts. `training/checkpoint.py` owns schema v1, exact preprocessing, restricted `weights_only=True` loading of safe primitive values, lists/mappings, and tensors, atomic save, resume identity, and environment metadata.

The four Faster R-CNN loss keys are produced by torchvision, not hardcoded by the trainer. The trainer can also record a different registered detector's loss mapping. Resume restores model, optimizer, optional scheduler, history, and RNG streams; it is not a shortcut for changing experiment semantics.

## Evaluation and prediction

`evaluation/metrics.py` adapts predictions/targets to torchmetrics and normalizes only negative backend sentinels to zero. `evaluation/errors.py` owns deterministic same-class greedy error labels. `evaluation/visualization.py` renders ordinary, difficult, and predicted boxes. `evaluation/evaluate.py` reconstructs from a checkpoint, verifies manifest identity, streams batches, writes JSON/CSV, reloads only ranked error images, and atomically publishes the output directory. `evaluation/comparison.py` reads existing run artifacts and reports compatible metric/config differences.

`inference/predictor.py` reconstructs architecture and ordered classes from a checkpoint with `weights="none"`. Single-image prediction protects both outputs from accidental overwrite and atomically writes its JSON, while the PNG renderer saves directly. Directory prediction stages a complete tree, records unreadable-image errors, and publishes it as a unit. It needs no YAML or manifest because it does not claim dataset metrics.

## Where to verify a change

Use `tests/test_end_to_end.py` for an executable package path and focused tests for the code you changed. A parser unit test does not cover model integration, a dry run does not measure learning, and a small run does not establish full VOC quality. The [Kaggle training record](../recorded-run/README.md) comes from a separate real execution; `configs/reference_fasterrcnn.yaml` alone is only a configuration.

Continue with [detection flow](detection-flow.md) for tensor ownership, [configuration flow](configuration-flow.md) for precedence, or [adding datasets](../guides/adding-datasets.md) and [adding models](../guides/adding-models.md) for internal extension checklists.
