# Configuration Flow and Ownership

[Simplified Chinese](configuration-flow.zh-CN.md) | [Field reference](../reference/config-reference.md)

This concept page is for readers tracing why a value was selected or why a command-line option is absent from saved configuration. It follows the implementation in `src/object_detector/config.py` and `src/object_detector/cli.py`.

## From text to strict `AppConfig`

```text
dataclass defaults
  -> recursively merge known YAML fields
  -> apply repeated --set dotted paths in command order
  -> construct dataclasses and Path values
  -> validate types, finiteness, ranges, and choices
  -> command-specific runtime arguments
  -> training preflight or command handler
```

There is no environment-variable configuration layer. Environment such as `TORCH_HOME` can affect torch's cache location, but it does not create an `AppConfig` source.

YAML must have a mapping root. All normal sections and fields are closed: an unknown key fails with its dotted path, and a scalar cannot replace a section. `model.params` is the one model-specific mapping; the selected model registry entry rejects reserved, misspelled, or unmaintained keys before construction.

Each `--set KEY VALUE` value is parsed as YAML. That makes `true`, `3`, `0.5`, `null`, lists, and mappings typed values. PyYAML keeps `none` as a string, while `null` and `~` become Python `None`. After merging, construction converts data/output paths to `Path`; validation rejects Booleans where integers or numbers are required, non-finite numbers, invalid ranges/choices, and empty identifiers.

## Inspect values and sources

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set train.epochs 3 --set model.params.min_size 320 --set run_name trace-me
```

Expected output is complete YAML with a final `sources` mapping. Every normal leaf is `default`, `yaml`, or `cli`. `model.params` is a leaf when empty; configured nested parameter paths are reported from YAML or CLI. The command only resolves configuration. It does not read manifests, choose a device, check output writability, inspect a weight cache, construct a model, or write artifacts.

The stored `artifacts/<run>/config.yaml` contains resolved values without the `sources` report. `run.yaml` separately records environment, resolved device, seed, Git revision, classes, and manifest identity.

## Training ownership

`detect train` loads `AppConfig` from `--config` and repeated `--set`. If command-level `--device` is present, `cli._train` uses `dataclasses.replace` after config validation. That final replacement is not visible in `show-config` source tracking, but the resulting value is written to the run's resolved `config.yaml`.

Then `training.run_training` loads `dataset.yaml`, calls `preflight.validate_training_request`, resolves `auto` to CUDA then MPS then CPU, seeds RNGs, constructs the registered model, and builds datasets/loaders. Preflight checks required manifest files, class count, requested accelerator availability, output destination writability, and whether a named backbone weight is cached. A missing weight is a notice, not an issue; download can occur later in torchvision model construction.

`--dry-run` and `--resume` change orchestration, not the recipe schema. Dry run consumes one training batch and writes no normal run artifacts. Resume points to a schema-versioned checkpoint and is checked against the resolved configuration.

## Runtime-only CLI arguments

| Command | Runtime-only inputs, not `AppConfig` leaves |
|---|---|
| `prepare-data` | `--data-dir`, `--manifest-dir`, `--allow-nonstandard-counts` (the paths happen to share config meanings but are parsed independently) |
| `inspect-data` | `--manifest-dir`, `--data-dir`, `--split`, `--limit` |
| `list-models` | none; reads registry metadata |
| `model-info` | positional model `name` |
| `compare-runs` | run directories, `--metric`, optional `--output` |
| `train` | `--config`, repeated `--set`, `--dry-run`, `--resume`, final `--device` |
| `evaluate` | `--checkpoint`, `--split`, `--output-dir`, `--device`, `--score-threshold`, `--overwrite` |
| `predict` | `--checkpoint`, exactly one of `--image`/`--input-dir`, `--output-dir`, `--device`, `--score-threshold`, `--display-limit`, `--overwrite` |

Evaluation has no `--config` or `--set`: it validates and loads the resolved config saved in the checkpoint for dataset paths, sample limit, error thresholds, maximum detections, batch size, and workers. Its CLI `--score-threshold` is a separate serialization/visualization threshold and does not replace the saved `evaluation.score_threshold` field in an `AppConfig`. Prediction uses checkpoint model/classes/preprocessing and runtime inputs only.

## Evidence and failure boundaries

Use `show-config` to prove text resolution, `train --dry-run` to prove config plus data/model/update integration, a bounded normal run to prove artifact publication, and evaluation to prove checkpoint plus matching manifest metrics. Passing an earlier boundary does not prove a later one.

Unknown fields or invalid types fail before model construction. Preflight issues fail before a normal run directory is created. Fresh training rejects an existing run directory. Checkpoint and text artifacts are individually atomic; evaluation and directory prediction stage and publish complete output directories. Continue with the [code tour](code-tour.md) for module ownership or the [configuration reference](../reference/config-reference.md) for every leaf.
