# Troubleshooting by Boundary

[Simplified Chinese](troubleshooting.zh-CN.md) | [Configuration reference](../reference/config-reference.md)

Use the smallest command that crosses the failing boundary. Preserve manifests, checkpoints, and existing output directories until the cause is known.

## Installation or parser failure

```bash
uv run detect --version
uv run detect show-config --config configs/learning_minimal.yaml
```

- `unknown configuration field: ...`: fix the exact YAML or `--set` path; unknown keys never pass through.
- `... must be ...`: YAML typed the value differently or it violates the documented range. Remember that `null` and `~` become Python `None`, while `none` remains a string.
- `invalid override`: `--set` values are parsed as YAML, so malformed YAML fails before training.
- Argparse `usage:` with exit 2: the option belongs to a different subcommand or has a missing/invalid value. For the training surface, check `uv run detect train --help`.

`show-config` reads no dataset, constructs no model, contacts no network, and writes no artifacts. Its `sources` mapping shows whether each leaf came from `default`, `yaml`, or `cli`.

## Data preparation or loading failure

- `... split has ... images; expected ...`: the tree is not complete official VOC 2007. Redownload or repair it. Use `--allow-nonstandard-counts` only for an intentional VOC-shaped fixture.
- `split contains duplicate image IDs` or `split overlap`: fix the split files; no partial manifest replacement occurs.
- `missing image`, `missing annotation`, filename mismatch, dimensions disagreement, invalid XML, unknown class, or nonpositive box: fix the named source sample, then rerun preparation.
- Training later reports an image-size mismatch: source content changed after preparation. Reprepare and use the new identity.

Inspect both structure and pixels:

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset-preview.png
```

Preparation and inspection do not create `dataset-preview.png`; only the preview script does. Its `--output` path is overwritten without prompting when that PNG already exists, so choose a new path when preserving earlier evidence. A successful parser does not prove that a custom coordinate convention looks correct.

## Preflight or model construction failure

- `missing train.csv, ...`: `data.manifest_dir` does not contain the required prepared files.
- `expected 21, dataset requires ...`: `model.expected_num_classes` must equal background plus metadata classes.
- `CUDA was requested but is unavailable` or `MPS was requested but is unavailable`: use an available device or fix the environment.
- `unsupported device`: use `auto`, `cpu`, `mps`, or a valid `cuda...` string.
- `cannot write below ...`: choose a writable `output_dir`.
- A notice says a weight is not cached: network is needed during construction unless the exact torchvision cache file is supplied. Choose `model.weights=none` for a guaranteed offline model path.
- `unknown model` or an unexpected keyword error: use `list-models` and `model-info`; correct `model.name` or `model.params`.

## Dry run or training failure

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

- `non-finite <loss> for image IDs [...]`: inspect the reported samples, coordinates, labels, learning rate, and full-precision CPU path. Do not silently skip the batch.
- Out-of-memory from the backend: lower `train.batch_size`, reduce model-owned input size through documented model parameters, or choose a different model. This changes the experiment and requires a new run.
- An existing run directory is rejected: choose a new `run_name`; a fresh run never appends to existing files.
- AMP expectations differ: scaling is enabled only on CUDA, and autocast is enabled only on CPU/CUDA, so a resolved MPS device always uses full precision. Current preflight prints the MPS notice only when configured `device` is exactly `mps`; `device=auto` can resolve to MPS without that notice.

Finite losses and `dry-run OK` prove one connected update, not convergence or detection quality.

## Resume, evaluation, or prediction failure

- `resume identity mismatch`: model name, ordered classes, manifest identity, or exact preprocessing changed. Start a new run or restore the matching inputs.
- `resume configuration changes training semantics`: only total epochs, worker count, device, output directory, and run name may change.
- Requested epochs are not greater than checkpoint epoch: increase `train.epochs`.
- Resume destination is unrelated and nonempty: point `run_name` at the checkpoint parent or use an empty new destination.
- Historical best checkpoint is unavailable or incompatible: when resuming `last.pt` into a different empty run, restore the matching sibling `best.pt`; alternatively resume directly from a valid `best.pt` into a new empty run, or use its exact original path in place only when `last.pt` is missing.
- Resume reports invalid metric history, a historical-best `lineage_id`, strict-best-history, or CUDA RNG mismatch: restore unedited lineage checkpoints. Every configured validation value must be finite and `best_metric` must equal the complete-history maximum. CUDA metadata must name an explicit device such as `cuda:0`, and its RNG entries are checked against that checkpoint's own `run_metadata.cuda_device_count`, not against `last.pt`.
- `unsupported schema_version`, restricted `weights_only=True` primitive/container/tensor load failure, or preprocessing-contract failure: the file is corrupt, untrusted, or not schema v1. Do not fall back to unrestricted pickle loading. Schema v1 intentionally contains safe primitive values, lists/mappings, and tensors.
- Evaluation reports manifest mismatch: restore the prepared data matching the checkpoint. Prediction can still run without manifests because it makes no dataset metric claim.
- Evaluation or prediction output exists: preserve it and choose a new path, or use `--overwrite` only after deciding replacement is intentional.

## Metrics look wrong

AP receives raw model predictions. Evaluation `--score-threshold` changes serialized predictions and images, not AP/AR. Error classification instead uses checkpoint configuration `evaluation.error_score_threshold` and `evaluation.error_iou_threshold`. Difficult targets are excluded from ordinary target counts and misses; their matching predictions become `ignored`.

Random initialization and bounded two-epoch runs may produce near-zero metrics. Inspect `evaluation.json`, `per_class.csv`, `errors.csv`, and `visualizations/` before forming a hypothesis. The reference recipe alone is not evidence; compare artifact structure with the [recorded run](../recorded-run/README.md), not just its score.

When reporting a failure, include the exact command, concise error, resolved config, manifest identity, checkpoint schema/hash when relevant, framework versions, device, and Git revision. Remove private paths and data. See the [code tour](../concepts/code-tour.md) for module ownership.
