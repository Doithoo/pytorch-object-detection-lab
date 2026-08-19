# Configuration Recipes

[Simplified Chinese](README.zh-CN.md) | [Configuration reference](../docs/reference/config-reference.md)

Configuration values resolve in this order: typed defaults, YAML, then repeated `--set KEY VALUE` overrides. A runtime `--device` override is applied last where the command supports it. Inspect the result before building a model:

```bash
uv run detect show-config --config configs/learning_minimal.yaml
```

This prints resolved YAML and value sources. It does not load data, construct a model, contact the network, or write training artifacts.

## Shipped recipes

| File | Role and scope | Network behavior | Expected training artifacts |
|---|---|---|---|
| `learning_minimal.yaml` | Default learning route using Faster R-CNN MobileNet V3 Large 320 FPN, 2 epochs, and train/valid/test limits of 32/16/16 | Offline model construction with `weights: none`; source data must already be local | A bounded run under `artifacts/run` unless `run_name` is overridden, containing `config.yaml`, `run.yaml`, `metrics.csv`, `best.pt`, and `last.pt` |
| `fasterrcnn_resnet50_fpn.yaml` | Short unbounded comparison recipe for Faster R-CNN ResNet-50 FPN; omitted fields inherit typed defaults | Offline model construction with `weights: none`; no dataset download occurs | The standard run artifact set; because there are no sample limits, do not mistake its 2 epochs for the bounded learning recipe or for an evidence-complete reference run |
| `ssdlite320_mobilenet_v3.yaml` | Short unbounded comparison recipe for SSDLite 320 MobileNet V3 Large; omitted fields inherit typed defaults | Offline model construction with `weights: none`; no dataset download occurs | The standard run artifact set, suitable for a controlled model-family comparison after assigning a unique `run_name` |
| `reference_fasterrcnn.yaml` | Full VOC reference recipe using Faster R-CNN MobileNet V3 Large 320 FPN, 26 epochs, step scheduling, and no sample limits; the checked-in defaults are CPU-safe | `weights: imagenet1k_v1` needs the pinned backbone weight in the torch cache or network access to download it | `artifacts/reference-fasterrcnn` with the standard run files; the recorded Kaggle run separately preserves its CUDA/AMP overrides and evaluation artifacts |

## Choosing a recipe

Use `learning_minimal.yaml` to learn `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`. Its sample limits make it a bounded learning run. Use either short model-family recipe only after inspecting resolved defaults and giving the run a unique name, for example:

```bash
uv run detect train --config configs/ssdlite320_mobilenet_v3.yaml --set run_name ssdlite-check --dry-run --device cpu
```

Expected dry-run output is batch diagnostics, named finite losses, and `dry-run OK`; no run directory or checkpoint is written.

Use `reference_fasterrcnn.yaml` only when the official prepared data and compute budget are ready. One evidence-complete execution is published in the [recorded run](../docs/recorded-run/README.md): its Kaggle runner changed operational fields to CUDA, AMP, two workers, and Kaggle paths, while preserving the model and optimization recipe. A YAML file by itself is still not result evidence.

## Artifact and comparison rules

A normal training command writes the resolved configuration rather than merely copying the input YAML. Preserve `config.yaml`, `run.yaml`, `metrics.csv`, and both checkpoints together. Assign a distinct `run_name` to avoid mixing experiments. `detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95` accepts only compatible run directories with matching manifest identities:

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

The command prints a comparison table and writes the optional CSV. It does not train, evaluate the reserved test split, or make two differently prepared datasets comparable.
