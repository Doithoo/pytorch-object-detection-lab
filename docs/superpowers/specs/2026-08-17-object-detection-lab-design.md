# PyTorch Object Detection Lab Design

## Summary

Create `pytorch-object-detection-lab` as the third independent repository in the
PyTorch image-learning series. The repository teaches a complete, reproducible
object-detection workflow with Pascal VOC 2007 and a torchvision Faster R-CNN
MobileNetV3-Large 320 FPN baseline.

The first release is focused but complete: data download and validation, fixed
manifests, dataset inspection, a real dry run, training and resume, COCO-style
evaluation, error visualization, checkpoint-only prediction, bilingual learning
material, offline tests, CI, and distributable packaging.

The repository name is `pytorch-object-detection-lab`, the Python package is
`object_detector`, and the console command is `detect`.

## Goals

- Teach learners who understand basic tensors, losses, and gradients how an
  object-detection system works from data preparation through error analysis.
- Preserve the series workflow:
  `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`.
- Make runs reproducible with official splits, immutable manifests, resolved
  configuration, content hashes, run metadata, and self-contained checkpoints.
- Keep the training and evaluation path readable while relying on torchvision
  for established detector implementations and a maintained metric library for
  average precision.
- Support a CPU-capable, fully offline learning configuration and a separate
  reference configuration with an explicitly versioned ImageNet-pretrained
  backbone.
- Ship a repository that can be linted, tested, built, and installed without
  downloading datasets or model weights during tests.

## Non-goals

The first release does not include COCO dataset support, instance segmentation,
tracking, rotated boxes, keypoints, ONNX export, a web demo, distributed
training, or a published full-VOC reference score. A full reference run will be
added only after its configuration, environment, artifacts, and metrics have
been recorded and verified. The design does not introduce a shared package or
monorepo coupling among the classification, segmentation, and detection labs.

## Audience And Learning Path

The primary reader has completed an introductory PyTorch exercise but has not
yet built an object detector. The guided path should take roughly 8-12 hours.
The first pass keeps one concept visible at a time and explains:

1. how variable numbers of boxes change dataset and batching contracts;
2. how VOC XML annotations become tensors;
3. why a detector returns losses in training mode and predictions in evaluation
   mode;
4. how region proposals, classification, and box regression fit together;
5. why AP is different from image-classification accuracy;
6. how confidence thresholds, IoU, missed objects, and false positives affect
   qualitative and quantitative results.

## Architectural Approach

Build an independent, native PyTorch/torchvision project. Reuse conventions from
the sibling labs rather than copying segmentation-specific code. torchvision
owns the detector implementations; the repository owns the explicit data,
configuration, training, checkpoint, evaluation, inference, and teaching
layers.

Do not base the learning path on MMDetection, Ultralytics, Lightning, or another
high-level training framework. Those frameworks provide broader functionality,
but they hide the data and loss flow this repository is intended to teach.

The package structure is:

```text
src/object_detector/
|-- cli.py
|-- config.py
|-- preflight.py
|-- data/
|   |-- schema.py
|   |-- voc.py
|   |-- manifest.py
|   |-- dataset.py
|   |-- transforms.py
|   `-- inspection.py
|-- models/
|   |-- spec.py
|   |-- registry.py
|   `-- torchvision_models.py
|-- training/
|   |-- train.py
|   |-- trainer.py
|   `-- checkpoint.py
|-- evaluation/
|   |-- metrics.py
|   |-- evaluate.py
|   |-- errors.py
|   `-- visualization.py
`-- inference/
    `-- predictor.py
```

Each module communicates through typed configuration objects, sample targets,
model specifications, or serialized result schemas. Dataset providers do not
construct models, models do not read manifests, and prediction does not require
the original training YAML.

## Commands

The installed CLI exposes:

- `detect --version`
- `detect show-config --config PATH [--set KEY VALUE ...]`
- `detect prepare-data --data-dir PATH --manifest-dir PATH`
- `detect train --config PATH [--dry-run] [--resume CHECKPOINT]`
- `detect evaluate --checkpoint CHECKPOINT --split {train,valid,test}`
- `detect predict --checkpoint CHECKPOINT (--image PATH | --input-dir PATH)
  --output-dir PATH`

Dataset download and preview remain explicit scripts, consistent with the
sibling repositories:

- `scripts/download_data.py` downloads and verifies VOC archives;
- `scripts/preview_dataset.py` renders images, boxes, labels, and difficult
  objects before training;
- `scripts/plot_metrics.py` renders recorded training curves.

Command functions are thin adapters. Domain behavior belongs in package
modules so it can be imported, tested, and reused.

## Configuration

Configuration precedence is:

```text
typed code defaults < YAML < --set overrides < dedicated CLI flags
```

The schema has top-level `data`, `model`, `train`, `evaluation`, `device`,
`output_dir`, and `run_name` sections. `show-config` prints the resolved values
and their source. Unknown fields, invalid types, and invalid combinations fail
before data or model construction.

The class count and label mapping come from prepared dataset metadata. A model
configuration may declare `expected_num_classes`, but only as a guard against
using the wrong dataset. Background is model label `0`; the 20 VOC categories
use stable labels `1` through `20` in canonical VOC order.

`configs/learning_minimal.yaml` uses the default Faster R-CNN model, no
downloaded weights, `num_workers: 0`, bounded sample counts, and a short run.
It must perform its dry run without network access. The dry run executes a real
batch load, model forward pass, loss sum, backward pass, and optimizer update.

`configs/reference_fasterrcnn.yaml` uses the same detector family with
`MobileNet_V3_Large_Weights.IMAGENET1K_V1` as its pinned ImageNet backbone
weight. Preflight reports whether that weight is cached and explains the network
requirement before model construction. The configuration is provided as a
reproducible recipe, not as an unverified benchmark claim.

Additional model configurations demonstrate Faster R-CNN ResNet50 FPN and
SSDLite MobileNetV3 without changing the main learning path.

## Pascal VOC Data

The source dataset is Pascal VOC 2007. The downloader obtains the official
train/validation and test archives, verifies known checksums, and extracts into
`data/raw/VOCdevkit/VOC2007` without silently overwriting unrelated content.

Preparation uses the official split files:

- train: 2,501 images;
- validation: 2,510 images;
- test: 4,952 images.

It verifies every image and annotation, checks that split membership is
disjoint, rejects duplicate IDs and unknown classes, and records relative paths
only. It writes `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`,
`source.yaml`, and a human-readable summary. Metadata includes class order,
source archive checksums, split membership hashes, preparation version, and
coordinate convention.

VOC XML box coordinates are one-based and inclusive. Preparation converts each
box to zero-based continuous `xyxy` coordinates as
`[xmin - 1, ymin - 1, xmax, ymax]`, clips it to image bounds, and rejects boxes
that are non-finite or have non-positive width or height after clipping. This
rule is recorded in dataset metadata and covered by boundary tests.

The canonical in-memory sample contract is:

```text
image: FloatTensor[C, H, W]
target:
  boxes: FloatTensor[N, 4]      # zero-based xyxy
  labels: Int64Tensor[N]
  image_id: Int64Tensor[1]
  area: FloatTensor[N]
  iscrowd: Int64Tensor[N]
  difficult: BoolTensor[N]
```

An image with no ordinary targets is valid and uses tensors with leading size
zero. During training, difficult objects are omitted from model targets. During
evaluation they remain available to the evaluator as ignored targets, so a
prediction matched to a difficult annotation is not counted as an ordinary
true positive or false positive. The evaluator adapter maps difficult targets to
`iscrowd: 1` and ordinary targets to `iscrowd: 0` for the COCO evaluator and
tests that difficult matches are ignored.

Transforms receive the image and target together. The first release supports
tensor conversion, optional horizontal flip, and image-only photometric
augmentation. Detector-owned resizing and normalization remain inside the
torchvision model. The first release does not add random crop because its box
retention policy would distract from the primary learning path.

The DataLoader uses an explicit detection collate function and returns lists of
images and targets rather than stacking a variable number of boxes.

## Models

The model registry exposes a small `ModelSpec` interface containing the stable
name, constructor, weight policy, preprocessing metadata, and capability flags.
It initially registers:

- `fasterrcnn_mobilenet_v3_large_320_fpn` as the default;
- `fasterrcnn_resnet50_fpn` as the heavier two-stage comparison;
- `ssdlite320_mobilenet_v3_large` as the one-stage comparison.

Constructors use public torchvision APIs and replace prediction heads for the
prepared dataset class count. Weight policy is explicit: `none` or a named,
pinned torchvision backbone enum. The learning configuration never converts an
unavailable weight request into random initialization silently.

The package does not reimplement region proposals, ROI Align, matching, box
regression, or non-maximum suppression. Teaching material explains these parts
and traces their observable inputs and outputs through torchvision.

## Training And Checkpoints

In training mode, a detector returns a dictionary of named losses. The trainer
logs each component and computes the optimization loss as their sum. It rejects
non-finite losses with the batch image IDs and component values included in the
error. Gradient clipping and AMP are optional and disabled in the minimal
configuration.

Each epoch writes aggregate training losses and validation metrics to
`metrics.csv`. The best checkpoint is selected by validation `map_50_95`; the
official test split is reserved for final evaluation and never participates in
checkpoint selection.

Checkpoint writes are atomic. A checkpoint contains:

- checkpoint schema version;
- resolved configuration;
- model specification and weight policy;
- ordered class names and label mapping;
- preprocessing metadata;
- model, optimizer, and scheduler state;
- epoch, best metric, and metric history;
- manifest identity and split hashes;
- Python, PyTorch, torchvision, platform, device, seed, and source revision
  metadata.

`last.pt` supports recovery after interruption. Resume validates model identity,
class mapping, preprocessing, and dataset hashes. It rejects semantic changes
and permits only explicit operational changes such as total epochs, device,
worker count, log level, and output location.

## Evaluation And Error Analysis

Use a maintained AP implementation compatible with COCO-style bounding-box
metrics, backed by `torchmetrics.detection.MeanAveragePrecision` and its
`pycocotools` backend. Both are normal locked project dependencies. Do not
hand-roll AP integration or IoU threshold averaging.

The stable evaluation report includes:

- `map_50_95`, `map_50`, and `map_75`;
- `mar_1`, `mar_10`, and `mar_100`;
- per-class AP@[.50:.95] and AR at 100 detections;
- evaluated image count, target count, prediction count, score threshold, and
  metric backend versions;
- manifest and checkpoint identities.

If a model returns no detections, evaluation still writes a schema-complete
zero-valued report. Per-image prediction records are written as JSON using
image IDs, class names, scores, and `xyxy` boxes.

Error analysis ranks high-confidence false positives, missed targets, and
low-IoU matches using documented IoU and score thresholds. It writes tabular
records and representative annotated images, including worst cases. Rendering
never changes the machine-readable predictions.

## Prediction

Prediction reconstructs the model, class mapping, and preprocessing solely from
the checkpoint. The user supplies exactly one image or one input directory, an
output directory, and optional score and display limits. Single-image mode
writes one JSON prediction file and one annotated image. Batch mode writes one
aggregate JSON file plus one annotated image per readable input image, using the
same per-image result schema.

Checkpoint compatibility is checked before loading state tensors. Unsupported
schema versions, mismatched model specifications, corrupt images, and unwritable
destinations produce concise, actionable errors.

## Run Artifacts

A normal run writes:

```text
artifacts/<run-name>/
|-- config.yaml
|-- run.yaml
|-- metrics.csv
|-- last.pt
|-- best.pt
|-- evaluation.json
|-- per_class.csv
|-- predictions.json
`-- visualizations/
```

`config.yaml` is the resolved configuration. `run.yaml` records immutable run
identity and environment metadata. Evaluation and prediction commands do not
overwrite previous results without an explicit destination or overwrite flag.

## Error Handling And Preflight

Preparation completes all validation before atomically replacing manifests.
Training preflight verifies split existence and non-overlap, hashes, class
metadata, detector compatibility, device availability, writable output paths,
and weight availability. It reports all independent validation failures in one
pass when practical.

Degenerate boxes created by a supported transform are filtered and counted.
Malformed source boxes fail preparation instead of being silently repaired.
Empty targets are accepted. Missing predictions are accepted. Corrupt
checkpoints and non-finite training loss fail immediately with diagnostic
context.

Errors distinguish user input problems, incompatible artifacts, unavailable
optional dependencies, and unexpected internal failures. CLI commands return a
nonzero exit status for failures and do not leave partially written final
artifacts.

## Documentation

README and core documentation are maintained in English and Simplified Chinese.
The structure follows the more mature segmentation lab:

- a short README workflow and repository map;
- a five-chapter tutorial plus an ordered learning path;
- concepts for the code tour, detection data flow, and how Faster R-CNN works;
- guides for custom VOC-style data, experiments, troubleshooting, adding data
  providers, and adding model adapters;
- references for configuration, dataset formats, model registry, metrics, and
  checkpoint schemas.

README visuals must come from an actually executed bounded VOC run, identify
their source, and remain separate from any claim of full-dataset reference
quality. Documentation must state clearly that the initial reference
configuration is a recipe without a published full-VOC score.

## Testing And CI

All automated tests are offline. Synthetic VOC fixtures contain multiple
objects, empty images, difficult annotations, boundary boxes, and malformed XML
cases. No test downloads VOC archives or pretrained weights.

Unit tests cover configuration resolution, XML parsing, coordinate conversion,
manifest identity, paired transforms, collate behavior, model registry,
checkpoint compatibility, evaluator input conversion, prediction serialization,
error ranking, and visualization.

Integration tests cover mixed target counts, difficult-object handling, atomic
artifact writes, interrupted-run recovery, and checkpoint-only inference. A
fast injectable detector completes the end-to-end train, resume, evaluate, and
predict workflow. A separate default torchvision-model smoke test performs one
real forward pass, loss calculation, backward pass, and optimizer update with
no pretrained weights.

CI runs on Python 3.10, 3.11, and 3.12 and performs Ruff linting, Ruff format
checking, mypy, pytest, package build, Twine metadata validation, and a clean
wheel smoke test. The wheel smoke test runs `detect --version` and
`detect show-config` without repository-relative imports.

## Release Criteria

The first release is ready when:

1. a clean Python 3.10-3.12 environment can install the locked development
   environment and the built wheel;
2. the official VOC 2007 archives can be verified and prepared into stable,
   disjoint manifests;
3. dataset preview exposes box and label mistakes before training;
4. the offline minimal dry run proves data loading, forward, loss, backward,
   and optimizer update are connected;
5. a bounded sample run can train, resume, evaluate, and predict;
6. evaluation emits complete metric, class, prediction, and visualization
   artifacts, including when there are no detections;
7. prediction reconstructs all training semantics from the checkpoint;
8. English and Chinese primary workflows contain tested commands;
9. lint, type, test, packaging, and clean-wheel CI checks pass without dataset
   or weight downloads.

Publishing a full VOC reference metric is deliberately a later milestone. It
requires a recorded configuration, environment, source revision, manifest
hashes, curves, class metrics, error cases, checkpoint hash, and an explicit
account of the ImageNet backbone initialization.
