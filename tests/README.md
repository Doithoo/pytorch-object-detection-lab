# Test Guide

[Simplified Chinese](README.zh-CN.md) | [Contributing](../CONTRIBUTING.md)

All repository tests must run offline. They use synthetic tensors, temporary VOC-shaped fixtures, fake detectors, or models constructed with `weights: none`. Install the development environment first with `uv sync --locked --extra dev`.

## Choose the smallest layer

| Layer | Use it when changing | Representative files | Expected output |
|---|---|---|---|
| Unit | One schema, transform, parser, metric, or checkpoint rule | `test_config.py`, `test_voc.py`, `test_transforms.py`, `test_metrics.py`, `test_checkpoint.py` | Fast pass/fail result for one behavior; temporary outputs only |
| Data integration | Preparation, manifest identity, loading, inspection, or preflight | `test_manifest.py`, `test_dataset.py`, `test_inspection.py`, `test_preflight.py` | Synthetic VOC manifests, targets, summaries, and previews under pytest temporary directories |
| Model and optimization | Registry metadata, weight policy, one detector update, or trainer behavior | `test_models.py`, `test_model_smoke.py`, `test_trainer.py`, `test_training.py` | Offline construction and synthetic or fixture-backed updates with finite loss values; no benchmark metric claim |
| Evaluation and inference | AP calculations, error analysis, run comparison, reports, or prediction | `test_evaluation.py`, `test_errors.py`, `test_comparison.py`, `test_inference.py` | Deterministic JSON, CSV, checkpoint, and image artifacts in temporary directories |
| End-to-end | A change affects CLI, data, training, evaluation, and prediction together | `test_end_to_end.py`, `test_cli.py` | One complete offline synthetic workflow and real parser behavior |
| Publication and examples | Documentation, package metadata, scripts, or runnable examples | `test_documentation.py`, `test_examples.py`, `test_packaging.py`, `test_scripts.py`, `test_download_data.py` | Valid links and commands, declared package files, executable offline help, and expected script outputs |

## Focused commands

Run one test while changing one behavior:

```bash
uv run --no-sync pytest tests/test_config.py::test_yaml_then_cli_override_precedence -q
```

Run a related group when several connected modules change:

```bash
uv run --no-sync pytest tests/test_manifest.py tests/test_dataset.py tests/test_inspection.py -q
uv run --no-sync pytest tests/test_models.py tests/test_model_smoke.py tests/test_trainer.py -q
uv run --no-sync pytest tests/test_evaluation.py tests/test_inference.py tests/test_comparison.py -q
uv run --no-sync pytest tests/test_documentation.py tests/test_examples.py tests/test_packaging.py -q
```

Focused success means the selected behavior passes. It does not imply that unrelated layers or the full workflow pass. `test_model_smoke.py` constructs a real torchvision detector and can be slower than fake-detector unit tests, but it does not download weights.

## Full verification

Before submitting a cross-cutting change, run the full suite with deprecations promoted to errors:

```bash
uv run --no-sync pytest -W error::DeprecationWarning
```

Then run the static checks listed in [CONTRIBUTING.md](../CONTRIBUTING.md). The full test suite should leave repository data and artifact directories untouched because generated files belong in pytest temporary directories.

## Scope

Passing tests confirms the covered software behavior, including an offline synthetic end-to-end path. It does not show detector convergence or establish a full VOC score. The separate [Kaggle training record](../docs/recorded-run/README.md) comes from official prepared data and a real GPU run; tests do not recreate it.
