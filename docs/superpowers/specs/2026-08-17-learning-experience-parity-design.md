# Object Detection Learning Experience Parity Design

## Goal

Bring the object-detection lab to the same standard of learning experience,
discoverability, and publication evidence as the image-segmentation lab while
preserving the detector project's explicit Pascal VOC 2007 scope.

This work is complete only when a learner can discover the supported models and
data contract without reading internal modules, follow a substantial bilingual
tutorial from tensors through error analysis, run every documented local command,
and understand which evidence comes from synthetic, bounded, or full-dataset
runs.

## Evidence And Current Gaps

The existing detector implementation has strong offline tests, atomic artifacts,
checkpoint validation, and a complete train/evaluate/predict path. Its learning
surface does not yet expose that depth:

- most detector tutorial chapters contain only seven lines, while the matching
  segmentation chapters contain roughly 50 to 170 lines;
- the detector documentation index is a short list instead of a task-oriented
  navigation page;
- `using-your-data`, metrics, dataset format, checkpoint, and model reference
  pages state contracts but do not teach how to inspect or apply them;
- the model registry is only discoverable through Python source;
- prepared data can be previewed, but there is no CLI summary for counts, boxes,
  classes, difficult objects, or empty images;
- run artifacts can be read manually, but there is no guarded comparison tool;
- there is no architecture decision record, configuration/script/test index, or
  reproducible source for documentation images;
- no full Pascal VOC 2007 reference run is available in the current repository
  or workspace.

## Scope

### Documentation

The English and Simplified Chinese documentation will use the same information
architecture as the segmentation lab without copying segmentation-specific
claims:

- README: audience, first runnable path, dry-run interpretation, artifact tour,
  model choices, evidence boundary, next steps, and repository map;
- documentation index: task-oriented links for tutorials, guides, references,
  concepts, recorded evidence, and architecture decisions;
- tutorial: six substantial chapters plus a time-boxed learning path;
- guides: model selection, custom VOC-shaped data, controlled experiments,
  troubleshooting, adding models, and the boundary for adding providers;
- reference: configuration, VOC data format, model registry, metrics,
  checkpoint schema, and VOC 2007 protocol;
- concepts: code tour, configuration/data flow, detection flow, and Faster R-CNN
  mechanics;
- supporting indexes for `configs/`, `scripts/`, `examples/`, and `tests/`.

Chinese pages should read as natural Chinese teaching material. English pages
may be more compact, but both languages must preserve commands, contracts,
safety boundaries, and links.

### CLI Discoverability

Add four read-only commands:

```text
detect list-models
detect model-info MODEL
detect inspect-data --manifest-dir DIR --split SPLIT [--limit N]
detect compare-runs RUN_DIR... --metric COLUMN [--output FILE]
```

`list-models` and `model-info` expose stable registry metadata without loading
weights. `inspect-data` summarizes prepared manifest content and decoded target
statistics without mutating data. `compare-runs` reads immutable run artifacts,
rejects incompatible manifest identities by default, reports configuration
differences, and never claims that a higher number proves a universally better
model.

The commands stay within the current architecture. This design does not add
automatic plugin discovery or serialize external Python code in checkpoints.

### Examples And Visual Evidence

Keep the existing progressive examples and add focused probes only where they
close a learning gap. Documentation images will be generated reproducibly from
synthetic inputs using the repository's actual box and error-visualization code.
Their captions must call them synthetic teaching diagrams, not model results.

A full VOC result may be published only after the recorded-run gate contains the
real resolved config, environment, manifest hashes, metrics, checkpoint hash,
runtime, and source image IDs. Dataset download and a long reference training run
remain a separate evidence-producing operation; the absence of that run must not
block honest learning documentation or be concealed by fabricated numbers.

## Architecture

### Model Metadata

Extend `ModelSpec` with user-facing metadata such as a short description,
supported parameters, and input notes. Registry lookup should provide a close
match for typos. Metadata is declarative and must not construct a detector or
download weights.

### Data Inspection

The inspection service reads `dataset.yaml` and one selected manifest, then
decodes up to the requested limit through `VocDetectionDataset`. It returns a
plain serializable report containing:

- dataset identity, split, total rows, and inspected rows;
- class counts for ordinary and difficult objects;
- empty-image and difficult-image counts;
- box width, height, and area ranges;
- image height and width ranges.

The CLI renders YAML so the output is both readable and easy to preserve in an
issue report. A limit of zero is rejected because it cannot support the stated
inspection contract.

### Run Comparison

The comparison service reads each run's `config.yaml`, `run.yaml`, and
`metrics.csv`. It validates the requested metric, chooses each run's best row
according to whether the metric is a loss, and requires equal manifest identity.
Incompatible runs fail with an explanation so learners do not accidentally rank
different data.

The report includes run name, model, best epoch, metric value, device, and
differing resolved configuration keys. Optional CSV output uses atomic
publication.

### Documentation Verification

Tests will enforce the parts most likely to decay:

- every primary English page has a Chinese counterpart and vice versa;
- relative Markdown links and referenced local images exist;
- documented `detect` commands use real parser options;
- documented Python entry points and configuration files exist;
- registered model names mentioned in reference tables are real;
- generated documentation assets are non-empty PNG files;
- new CLI commands and report services have focused offline tests.

Tests will not enforce arbitrary prose line counts. Depth is established by the
required topic checklist, runnable examples, and human review rather than padding.

## Error Handling

All new commands follow the existing CLI contract: expected input errors produce
a concise `error:` message and exit code 2. Missing artifacts identify the exact
path. Unknown models include a likely spelling suggestion. Inspection rejects an
unknown split, empty manifest, or invalid limit before expensive model work.
Comparison identifies the missing metric columns and incompatible manifest IDs.

No command overwrites an existing output file unless the operation documents and
implements an explicit overwrite policy. Read-only commands do not alter runs,
manifests, checkpoints, or source data.

## Testing And Acceptance

Acceptance requires:

1. focused tests for model discovery, data inspection, run comparison, document
   links, bilingual pairs, and asset generation;
2. the complete offline test suite with deprecation warnings treated as errors;
3. Ruff lint and format checks, mypy, lock verification, and `git diff --check`;
4. direct execution of representative examples and all new CLI help paths;
5. fresh sdist/wheel build, Twine validation, and clean-wheel CLI smoke tests;
6. a final documentation audit showing that no primary tutorial, guide, concept,
   or reference page remains a placeholder;
7. explicit reporting that the full VOC score is absent unless a genuine recorded
   run has actually completed and passed its evidence gate.

## Non-Goals

- copying the segmentation lab's mask providers or segmentation model zoo;
- adding external `module:function` detector factories without a separate
  checkpoint portability design;
- automatic third-party plugin discovery;
- distributed training, multi-GPU training, or benchmark leaderboard claims;
- publishing synthetic or bounded-run numbers as a full VOC 2007 result;
- using the official VOC test split for model selection.
