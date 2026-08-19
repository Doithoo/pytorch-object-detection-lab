# Object Detection Learning Experience Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the detector lab as navigable, executable, and educationally
complete as the segmentation lab without claiming an unexecuted full VOC run.

**Architecture:** Add small read-only discovery and reporting services behind the
existing CLI, then teach those stable contracts through substantial bilingual
documentation. Generate visual teaching assets from deterministic synthetic data
and protect commands, links, language pairs, and assets with offline tests.

**Tech Stack:** Python 3.10-3.12, argparse, dataclasses, PyYAML, PyTorch,
torchvision, Pillow, pytest, Ruff, mypy, uv/build/Twine.

**Commit policy:** The user did not request commits. Complete and verify the work
in the current worktree without committing.

---

### Task 1: Expose Model Registry Metadata

**Files:**
- Modify: `src/object_detector/models/spec.py`
- Modify: `src/object_detector/models/registry.py`
- Modify: `src/object_detector/cli.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Add failing registry and CLI tests**

Assert that registry entries expose a description, parameters, and input notes;
that a misspelled name suggests the nearest model; and that `list-models` and
`model-info` return metadata without calling a constructor.

```python
def test_unknown_model_suggests_close_match() -> None:
    with pytest.raises(ModelConfigError, match="did you mean 'fasterrcnn_resnet50_fpn'"):
        get_model_spec("fasterrcnn_resnet50_fp")
```

Run:

```bash
uv run pytest tests/test_models.py tests/test_cli.py -q
```

Expected: the new tests fail because the metadata and commands do not exist.

- [x] **Step 2: Implement declarative model metadata and discovery commands**

Add immutable fields to `ModelSpec` and populate all three built-in models. Use
`difflib.get_close_matches` for typo hints. Render stable tabular output for
`list-models` and labeled YAML-like lines for `model-info`; do not construct the
model or resolve weight downloads.

- [x] **Step 3: Re-run focused tests**

Run the same focused command and expect all tests to pass.

### Task 2: Add Prepared-Data Inspection

**Files:**
- Modify: `src/object_detector/data/inspection.py`
- Modify: `src/object_detector/cli.py`
- Modify: `tests/test_inspection.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Add failing report tests**

Use the synthetic VOC fixture to assert identity, total/inspected counts, ordinary
and difficult class counts, empty-image counts, and box/image ranges. Cover an
invalid zero limit.

```python
report = inspect_prepared_data(
    prepared_voc.manifests,
    split="valid",
    data_dir=prepared_voc.data_dir,
    limit=2,
)
assert report["split"] == "valid"
assert report["inspected_images"] == 2
```

- [x] **Step 2: Implement the serializable inspection service**

Read metadata and manifest rows first, reject empty or invalid requests, load
samples with `training=False`, and aggregate deterministic plain Python values.
Keep rendering helpers in the same module but separate from aggregation logic.

- [x] **Step 3: Add `detect inspect-data`**

Accept `--manifest-dir`, optional `--data-dir`, `--split`, and positive `--limit`.
Print the report as stable YAML and preserve the CLI exit-code-2 error contract.

- [x] **Step 4: Verify focused behavior**

```bash
uv run pytest tests/test_inspection.py tests/test_cli.py -q
```

### Task 3: Add Guarded Run Comparison

**Files:**
- Create: `src/object_detector/evaluation/comparison.py`
- Modify: `src/object_detector/evaluation/__init__.py`
- Modify: `src/object_detector/cli.py`
- Create: `tests/test_comparison.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Add failing comparison tests**

Create two minimal run directories containing resolved `config.yaml`, `run.yaml`,
and `metrics.csv`. Assert best-row selection, descending metric ordering, loss
ordering, flattened config differences, missing-column errors, manifest mismatch
errors, and existing-output rejection.

- [x] **Step 2: Implement immutable report types and readers**

Define frozen `ComparisonRow` and `ComparisonReport` dataclasses. Parse artifacts
with structured YAML/CSV APIs, validate finite numeric metric values, and include
the exact missing path or column in failures.

- [x] **Step 3: Implement text and atomic CSV rendering**

Keep comparison factual: run, model, epoch, selected value, device, and resolved
configuration differences. Write optional CSV through a sibling temporary file
and `Path.replace`, rejecting an existing destination.

- [x] **Step 4: Add `detect compare-runs` and verify**

```bash
uv run pytest tests/test_comparison.py tests/test_cli.py -q
```

### Task 4: Generate Reproducible Documentation Assets

**Files:**
- Create: `scripts/generate_doc_assets.py`
- Create: `docs/assets/detection-target-anatomy.png`
- Create: `docs/assets/detection-error-analysis.png`
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_documentation.py`

- [x] **Step 1: Add failing script and PNG tests**

Import the script without side effects, generate into a temporary directory, and
verify both PNGs open with Pillow, have stable dimensions, and contain more than
one color.

- [x] **Step 2: Generate assets through project rendering code**

Build deterministic synthetic RGB tensors, targets, and predictions. Use
`draw_detections` and `render_detection_evidence` rather than drawing result boxes
through a second undocumented implementation. Add visible titles/captions stating
that the images are synthetic teaching diagrams.

- [x] **Step 3: Run the generator and focused tests**

```bash
uv run python scripts/generate_doc_assets.py --output-dir docs/assets
uv run pytest tests/test_scripts.py tests/test_documentation.py -q
```

### Task 5: Strengthen Documentation Contracts

**Files:**
- Modify: `tests/test_documentation.py`
- Modify: `tests/test_examples.py`

- [x] **Step 1: Add bilingual-pair and relative-link checks**

Scan primary `docs/` sections while excluding internal `superpowers/` plans.
Require `.md`/`.zh-CN.md` pairs, resolve relative Markdown targets (including
anchors by validating the file component), and require local image files.

- [x] **Step 2: Validate documented configs and registry names**

Check `uv run python` paths, `--config` targets, and model names in the model-zoo
reference against parser and registry APIs.

- [x] **Step 3: Verify the test itself catches a broken fixture**

Use parameterized helper-level tests for missing links and language pairs, then:

```bash
uv run pytest tests/test_documentation.py tests/test_examples.py -q
```

### Task 6: Rebuild Navigation And Supporting Indexes

**Files:**
- Modify: `docs/README.md`
- Modify: `docs/README.zh-CN.md`
- Modify: `docs/tutorial/README.md`
- Modify: `docs/tutorial/README.zh-CN.md`
- Modify: `examples/README.md`
- Modify: `examples/README.zh-CN.md`
- Create: `configs/README.md`
- Create: `configs/README.zh-CN.md`
- Create: `scripts/README.md`
- Create: `scripts/README.zh-CN.md`
- Create: `tests/README.md`
- Create: `tests/README.zh-CN.md`

- [x] **Step 1: Replace list-only indexes with task-oriented navigation**

Explain where to start, when to use each guide/reference/concept, prerequisites,
expected outputs, and evidence boundaries. Link only to files that exist.

- [x] **Step 2: Document configuration, script, example, and test responsibilities**

For every shipped configuration and script, state its role, network behavior, and
expected artifacts. Explain test layers and how to run focused versus full suites.

- [x] **Step 3: Run documentation tests**

```bash
uv run pytest tests/test_documentation.py tests/test_examples.py tests/test_packaging.py -q
```

### Task 7: Rewrite The Six-Chapter Tutorial And Learning Path

**Files:**
- Modify: `docs/tutorial/learning-path.md`
- Modify: `docs/tutorial/learning-path.zh-CN.md`
- Modify: `docs/tutorial/00-basics.md`
- Modify: `docs/tutorial/00-basics.zh-CN.md`
- Modify: `docs/tutorial/01-environment.md`
- Modify: `docs/tutorial/01-environment.zh-CN.md`
- Modify: `docs/tutorial/02-data-and-boxes.md`
- Modify: `docs/tutorial/02-data-and-boxes.zh-CN.md`
- Modify: `docs/tutorial/03-faster-rcnn.md`
- Modify: `docs/tutorial/03-faster-rcnn.zh-CN.md`
- Modify: `docs/tutorial/04-training.md`
- Modify: `docs/tutorial/04-training.zh-CN.md`
- Modify: `docs/tutorial/05-evaluation-and-inference.md`
- Modify: `docs/tutorial/05-evaluation-and-inference.zh-CN.md`

- [x] **Step 1: Teach tensors, coordinate conventions, and variable batches**

Include hand calculations, empty targets, foreground/background labels, and the
synthetic target image. Tie every concept to `examples/01` or `examples/02`.

- [x] **Step 2: Teach environment and data trust boundaries**

Cover locked installs, CPU/CUDA/MPS checks, offline weights, official archive
checksums, manifests, difficult objects, inspection reports, and visual previews.

- [x] **Step 3: Teach detector mode contracts and one optimization step**

Explain backbone/FPN/RPN/ROI heads through tensor responsibilities, then map the
four torchvision loss names to training behavior. Distinguish dry run, bounded
learning, and full training.

- [x] **Step 4: Teach evaluation and error analysis**

Explain AP/AR, thresholds, difficult targets, missed/false-positive evidence,
checkpoint-only inference, and how to inspect artifacts without selecting on test.

- [x] **Step 5: Verify all commands and links**

Run documentation tests and directly execute examples 01-03.

### Task 8: Expand Guides, References, Concepts, And ADR

**Files:**
- Create: `docs/guides/using-models.md`
- Create: `docs/guides/using-models.zh-CN.md`
- Modify: `docs/guides/adding-datasets.md`
- Modify: `docs/guides/adding-datasets.zh-CN.md`
- Modify: `docs/guides/adding-models.md`
- Modify: `docs/guides/adding-models.zh-CN.md`
- Modify: `docs/guides/experiments.md`
- Modify: `docs/guides/experiments.zh-CN.md`
- Modify: `docs/guides/troubleshooting.md`
- Modify: `docs/guides/troubleshooting.zh-CN.md`
- Modify: `docs/guides/using-your-data.md`
- Modify: `docs/guides/using-your-data.zh-CN.md`
- Modify: `docs/reference/checkpoint-schema.md`
- Modify: `docs/reference/checkpoint-schema.zh-CN.md`
- Modify: `docs/reference/config-reference.md`
- Modify: `docs/reference/config-reference.zh-CN.md`
- Modify: `docs/reference/dataset-format.md`
- Modify: `docs/reference/dataset-format.zh-CN.md`
- Modify: `docs/reference/metrics.md`
- Modify: `docs/reference/metrics.zh-CN.md`
- Modify: `docs/reference/model-zoo.md`
- Modify: `docs/reference/model-zoo.zh-CN.md`
- Modify: `docs/concepts/code-tour.md`
- Modify: `docs/concepts/code-tour.zh-CN.md`
- Modify: `docs/concepts/detection-flow.md`
- Modify: `docs/concepts/detection-flow.zh-CN.md`
- Modify: `docs/concepts/how-faster-rcnn-works.md`
- Modify: `docs/concepts/how-faster-rcnn-works.zh-CN.md`
- Create: `docs/concepts/configuration-flow.md`
- Create: `docs/concepts/configuration-flow.zh-CN.md`
- Create: `docs/reference/voc2007.md`
- Create: `docs/reference/voc2007.zh-CN.md`
- Create: `docs/architecture/0001-reproducible-voc-detection-contracts.md`
- Create: `docs/architecture/0001-reproducible-voc-detection-contracts.zh-CN.md`

- [x] **Step 1: Expand task guides with executable decision paths**

Cover selecting models, bringing VOC-shaped data, running one-variable
experiments, diagnosing failures, and extending internal registries without
claiming an external plugin API.

- [x] **Step 2: Turn references into complete contract tables**

List every config field/default/range, exact manifest and checkpoint fields,
metric semantics/sentinels, model parameters/weights, and VOC split protocol.

- [x] **Step 3: Expand concepts and record the accepted architecture**

Add configuration flow, deepen code/data/model flow explanations, and document
why fixed manifests, self-contained checkpoints, torchvision mode contracts, and
read-only discovery tools were chosen.

- [x] **Step 4: Verify bilingual links and commands**

```bash
uv run pytest tests/test_documentation.py -q
```

### Task 9: Rebuild The Project README

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [x] **Step 1: Add audience, runnable path, outputs, visuals, and next steps**

Use the generated synthetic target image with an explicit caption. Show the first
workflow, model discovery, data inspection, dry-run interpretation, artifact
reading, model choices, and documentation navigation.

- [x] **Step 2: Preserve evidence honesty**

State prominently that no complete VOC score is published. Do not add a metric,
runtime, or result image unless it came from an evidence-complete recorded run.

- [x] **Step 3: Run README command and link verification**

```bash
uv run pytest tests/test_documentation.py tests/test_packaging.py -q
```

### Task 10: Complete The Release Audit

**Files:**
- Modify only files required by failures found during verification.

- [x] **Step 1: Run focused new-feature tests**

```bash
uv run pytest tests/test_models.py tests/test_cli.py tests/test_inspection.py tests/test_comparison.py tests/test_documentation.py tests/test_scripts.py -q
```

- [x] **Step 2: Run the full quality gate**

```bash
uv run pytest -W error::DeprecationWarning
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv lock --check
git diff --check
```

- [x] **Step 3: Execute representative user paths**

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_resnet50_fpn
uv run python examples/03_model_contract.py
uv run python scripts/generate_doc_assets.py --output-dir /tmp/object-detector-doc-assets
```

- [x] **Step 4: Build and smoke test distributions**

Build fresh sdist and wheel, run Twine validation, install the wheel into a clean
temporary virtual environment, and execute `detect --version`, `list-models`, and
`model-info` from the installed entry point.

- [x] **Step 5: Audit completion against the design**

Inspect every primary documentation file, all referenced assets, CLI help, package
contents, and Git status. Report full VOC evidence as absent unless a genuine
reference run was separately completed.
