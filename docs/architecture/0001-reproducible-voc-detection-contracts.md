# ADR 0001: Reproducible Object Detection Contracts

[Simplified Chinese](0001-reproducible-voc-detection-contracts.zh-CN.md)

- Status: Accepted
- Audience: maintainers and contributors changing data, models, training, evaluation, or artifact compatibility

## Context

Object detection combines source images and annotations, split policy, coordinate conversion, variable-size batches, model mode changes, optimizer state, random state, metric backends, and visual outputs. The project keeps these boundaries explicit so that a saved result can be inspected and compared without relying on hidden local state.

The repository contains reproducible VOC 2007 material, custom VOC-shaped class support, a COCO JSON provider, five torchvision detector entries, explicit external model factories, checkpoint-only prediction, and a separate YOLO data exporter. A completed Kaggle run is kept as one concrete result; it is not a universal benchmark.

## Decisions

### Prepared data has an identity

Preparation validates source samples and split membership before publishing `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`, `source.yaml`, and `summary.txt`. Metadata records ordered classes, the label mapping, annotation format, split counts, source hashes, CSV hashes, coordinate convention, and a combined identity.

VOC-shaped data infers non-empty class names from validated XML. Official VOC 2007 retains its published class order. COCO JSON accepts sparse category IDs and writes a stable continuous label mapping. Runtime targets always use background label 0, foreground labels from 1, zero-based continuous `xyxy` boxes, aligned fields, and `iscrowd` semantics.

Training and evaluation verify the prepared metadata and source bytes before making a metric claim. The manifest references source files; it is not a portable copy of the dataset.

### Checkpoints are safe, explicit, and versioned

Schema version 1 stores resolved configuration, model name, optional explicit factory path, model parameters, weight policy, ordered classes, preprocessing, manifest identity, split hashes, model/optimizer/scheduler state, epoch, selected metric, metric history, environment metadata, and random states.

Loading uses `weights_only=True`. Checkpoints never serialize executable model code. Built-in models are reconstructed from the registry. External models record a `module:function` path and require that path to be importable when prediction, evaluation, or resume reconstructs the model.

### Models share a small detection interface

Models receive RGB float tensors in `[0,1]` as a list. Train mode receives aligned targets and returns scalar loss mappings. Eval mode receives images and returns per-image `boxes`, `labels`, and `scores`. The trainer and evaluator validate this interface while leaving detector internals to torchvision or an explicit external factory.

The registry currently includes Faster R-CNN MobileNet, Faster R-CNN ResNet-50, RetinaNet ResNet-50, FCOS ResNet-50, and SSDLite MobileNet. The model modification example shows how to replace a backbone and anchor generator while preserving the shared interface.

### Metrics state their definition

Evaluation records COCO-style AP/AR from torchmetrics and VOC 2007 eleven-point AP at IoU 0.5. Validation can select either `map_50_95` or `voc_map_50_11`. Test results are intended for a final fixed comparison, not repeated tuning.

### Discovery and publication are non-mutating by default

Configuration display, model listing, model metadata, data inspection, and run comparison do not construct models, download weights, or rewrite existing artifacts. Preparation, evaluation, directory prediction, and YOLO export stage output and publish atomically. Nonempty destinations require explicit overwrite behavior.

### YOLO is an export boundary

`export-yolo-data` converts a validated prepared dataset to normalized YOLO text labels and `data.yaml`. It does not replace the repository trainer with a third-party engine. Different YOLO implementations have different architectures, losses, checkpoints, result formats, dependencies, and licenses; their exact version and license must be reviewed separately.

## Consequences

Preparation reads source files and therefore costs I/O, but later runs can verify the same identity. Checkpoints are larger because they include optimizer, history, and random state, but prediction and resume do not depend on an unrecorded YAML file.

The shared detector interface makes architecture comparisons and model changes easy to inspect. It does not make numerical behavior identical across frameworks, hardware, or model-owned postprocessing. Results should record their environment, configuration, data identity, metric definition, and scope.

External factories increase flexibility while retaining an explicit import boundary. They also make the factory path and its dependencies part of experiment provenance. The YOLO exporter provides interoperability without pretending that separate training engines are the same system.

These decisions improve inspectability and portability; they do not turn a dry run, bounded experiment, or configuration file into a complete benchmark.

See the [configuration reference](../reference/config-reference.md), [dataset format](../reference/dataset-format.md), [checkpoint schema](../reference/checkpoint-schema.md), [model reference](../reference/model-zoo.md), and [metrics reference](../reference/metrics.md).
