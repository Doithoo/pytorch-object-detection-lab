# ADR 0001: Reproducible VOC Detection Contracts

[Simplified Chinese](0001-reproducible-voc-detection-contracts.zh-CN.md)

- Status: Accepted
- Date: 2026-08-17
- Audience: maintainers and contributors changing data, models, training, evaluation, or artifact compatibility

## Context

Object detection joins several stateful systems: source images and XML, split policy, coordinate conversion, variable-size batches, torchvision models whose API changes by mode, optimizer/RNG state, metric backends, and visual evidence. A short script can make these dependencies implicit, but then two runs with the same YAML may not mean the same experiment.

This repository is a learning project centered on Pascal VOC 2007. It needs an offline random-weight path, controlled model comparison, safe resume, checkpoint-only prediction, and artifacts whose scope can be audited. It must also distinguish synthetic or VOC-shaped fixtures from the official protocol. At the time of this decision, no complete VOC run existed and `configs/reference_fasterrcnn.yaml` was only a procedure. The later [recorded run](../recorded-run/README.md) satisfies the evidence gate without changing this architecture.

The project maintains three torchvision models and one VOC provider. It does not yet have enough stable provider/model diversity, migration machinery, or security policy to promise arbitrary external code loading.

## Decision

Adopt six explicit contracts.

### 1. Fixed, content-derived manifests

Preparation validates all source samples and split membership before training, then publishes fixed `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`, `source.yaml`, and `summary.txt`. Split hashes cover each row, relative path, and complete referenced JPEG/XML bytes. The combined identity covers dataset name, ordered foreground classes, coordinate convention, and split hashes.

Manifests reference source paths; they do not copy source data. Official preparation enforces 2501/2510/4952 train/valid/test counts. `--allow-nonstandard-counts` bypasses only those counts, leaving all structural/content checks in force and preventing nonstandard fixtures from being described as official results.

Run artifacts and checkpoints record manifest identity and split hashes. Resume and dataset evaluation require identity equality. Prediction does not require a manifest because it makes no labeled-dataset metric claim.

### 2. Self-contained, schema-versioned checkpoints

Schema version 1 stores resolved configuration, stable model name and parameters, training weight policy, ordered class names, an exact preprocessing contract, manifest identity and split hashes, model/optimizer/scheduler state, completed epoch, best metric, history, environment metadata, and Python/NumPy/torch/DataLoader RNG state.

Consumers load through `torch.load(..., weights_only=True)` and require exact schema/preprocessing. Evaluation and prediction rebuild the registered architecture with `weights="none"` and load saved state, so they do not redownload training backbone weights. Prediction needs no YAML. Resume allows only total epochs, worker count, device, output root, and run name to change; semantic changes start a new run.

### 3. Preserve torchvision's mode contract

Datasets return lists of float RGB tensors and aligned target mappings. Torchvision owns detector normalization, resizing, padding, backbone/FPN/RPN/ROI or one-stage internals, and postprocessing.

In train mode, models receive images and targets and return named scalar losses. In eval mode, they receive images and return boxes, labels, and scores. The trainer validates and sums the returned loss mapping rather than hiding it behind a project-specific output object. Evaluation consumes the prediction mapping directly.

### 4. Keep discovery, inspection, and comparison non-mutating by default

`show-config`, `list-models`, and `model-info` must not construct models, inspect/download weights, or write run artifacts. `inspect-data` reads a bounded prepared split without changing it. `compare-runs` reads existing run artifacts and never edits them; optional `--output` creates a separate new CSV and refuses to overwrite it.

These commands expose selected values, sources, registry metadata, manifest identity, and configuration differences before expensive or state-changing work. They report evidence and do not declare a universally best model.

### 5. Publish artifacts atomically

Manifest preparation, evaluation, and directory prediction stage complete directories beside their destinations and publish with `os.replace`, with backup restoration on replacement failure. Checkpoints and training YAML/CSV use temporary files plus replacement; prediction/evaluation JSON is also written atomically. A single-image prediction PNG is saved directly after overwrite checks, which is a known narrower integrity boundary. Directory evaluation and directory prediction reject nonempty destinations unless `--overwrite` is explicit. Single-image prediction instead checks only the selected image stem's JSON/PNG collisions, so unrelated files may already exist in its output directory. Fresh run directories are never reused.

For staged directory publication and individually atomic JSON/CSV/checkpoint files, partial work must not appear under the corresponding final name. The directly saved single-image PNG is the explicit exception. A completed artifact can still describe a bounded or failed-quality experiment, so atomicity is an integrity property, not a performance claim.

### 6. Keep extension internal until an external API is designed

Model constructors and `ModelSpec` entries live in the internal registry. Dataset support is fixed to `voc2007`; a genuinely different provider requires repository changes. Configuration cannot import arbitrary `module:function` factories, automatic entry points are not discovered, and checkpoints do not serialize executable user code.

New internal models must preserve offline `weights="none"`, pinned named backbone policies, mode behavior, checkpoint reconstruction, tests, and bilingual documentation. New internal datasets must preserve preparation/runtime separation, target tensors, split/identity rules, preflight, and evidence semantics. This is not a stable external plugin API.

## Alternatives Considered

### Scan raw data and split at each run

This reduces manifest code, but filesystem order, random splitting, late validation, and source mutation can change experiment meaning. It also makes checkpoint/data compatibility hard to prove. Rejected in favor of fixed prepared membership and content identity.

### Hash only paths, timestamps, or split IDs

This is faster than reading all bytes, but annotations or images could change without a reliable identity change, and timestamps can change without content changes. Rejected because content trust is more important than preparation speed for VOC-scale learning runs.

### Save model weights only and require the original YAML/code

This makes smaller, simpler checkpoints, but class order, preprocessing, architecture parameters, manifest identity, and resume state become external hidden dependencies. Rejected for project checkpoints. The consequence is larger files and a schema that must be migrated deliberately.

### Use unrestricted pickle to support arbitrary objects

This could preserve custom schedulers and factories with less schema work, but it can execute code during load and makes portability opaque. Rejected. Safe tensor/container loading limits what may enter checkpoints and requires explicit fields.

### Wrap train and eval into one uniform detector output

A wrapper could hide torchvision's mode change, but would obscure the API learners must understand and add adaptation risk around losses and predictions. Rejected. The project instead teaches the two modes and checks their outputs at the trainer/evaluator boundaries.

### Let inspection commands construct models or download missing resources

This could provide richer metadata, but listing a model might unexpectedly consume network, cache, memory, or time. Rejected. Metadata is registered explicitly; construction belongs to dry run/training.

### Write final directories and files incrementally

This is simpler and can expose progress, but interrupted preparation/evaluation could look complete and overwrite prior evidence. Rejected for published artifacts. Logs/progress may be transient, but contract files appear only after successful replacement.

### Support external factories or automatic plugins now

This would enable arbitrary models/classes sooner, but raises unresolved questions about import trust, dependency reporting, config validation, target/class schemas, checkpoint portability, versioning, and failure messages. Rejected for version 0.1. It can be reconsidered only with an explicit versioned API and security/compatibility decision.

## Consequences

Preparation reads every referenced image and XML to validate and hash them, so it costs I/O and must be repeated after source changes. Runtime still depends on source data at the recorded relative layout; the manifest is not a portable copy.

Checkpoints are larger because they include optimizer, history, and RNG state. They remain coupled to registered architecture code and schema version; removing or changing a registry entry requires migration or an intentional compatibility break. Safe loading improves the trust boundary but does not prove that tensor values are benign or that a checkpoint is scientifically valid.

The torchvision contract keeps implementation thin and examples transferable, but behavior can depend on pinned torch/torchvision versions and model-owned postprocessing. Backend versions and resolved environment are therefore recorded.

Atomic staging uses additional temporary disk space and same-parent replacement. Explicit overwrite and unique run names require more deliberate artifact management, but reduce accidental evidence loss. A crash during single-image PNG rendering can still leave an incomplete final PNG; directory prediction avoids that gap through staging.

The closed extension surface limits arbitrary-class custom datasets and external models today. Contributors must change core code and tests. This is intentional until the project can support a stable, safe, portable plugin contract.

These decisions make provenance and failure boundaries inspectable; they do not guarantee reproducible floating-point results across different hardware, worker scheduling, framework versions, or nondeterministic kernels. They also do not convert a dry run, bounded sample experiment, or recipe into a complete VOC result.

See [configuration flow](../concepts/configuration-flow.md), [dataset contract](../reference/dataset-format.md), [checkpoint schema](../reference/checkpoint-schema.md), and [metrics](../reference/metrics.md) for executable contracts.
