# PyTorch Object Detection Lab

[简体中文](README.zh-CN.md) | [Documentation](docs/README.md) | [Examples](examples/README.md)

A beginner-oriented, reproducible Pascal VOC 2007 object-detection laboratory built with PyTorch and torchvision. It is for learners who want to inspect every boundary around a maintained detector rather than copy a hidden training script.

The learning path is exactly: `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`.

## Install

Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --locked --extra dev
uv run detect --version
```

## Seven-stage workflow

1. Download the two official VOC 2007 archives and verify their published MD5 checksums.

   ```bash
   uv run python scripts/download_data.py --data-dir data/raw
   ```

2. Validate the official split files and write deterministic manifests.

   ```bash
   detect prepare-data --data-dir data/raw --manifest-dir data/manifests
   ```

3. Inspect images, converted boxes, labels, and difficult annotations before training.

   ```bash
   uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --output artifacts/dataset_preview.png
   ```

4. Run one offline forward/backward update. The default learning configuration requests no pretrained weights.

   ```bash
   detect train --config configs/learning_minimal.yaml --dry-run --device cpu
   ```

5. Run a bounded learning experiment. Remove the sample limits only after the complete workflow is understood.

   ```bash
   detect train --config configs/learning_minimal.yaml --set train.epochs 2 --device cpu
   ```

6. Evaluate a self-contained checkpoint and write COCO-style metrics, predictions, deterministic errors, and evidence images.

   ```bash
   detect evaluate --checkpoint artifacts/run/best.pt --split test --output-dir artifacts/evaluation --device cpu
   ```

7. Predict one image or a directory without a YAML file or weight download.

   ```bash
   detect predict --checkpoint artifacts/run/best.pt --image data/raw/VOCdevkit/VOC2007/JPEGImages/000001.jpg --output-dir artifacts/prediction --device cpu
   ```

## Artifacts

A training run contains the resolved `config.yaml`, environment and manifest provenance in `run.yaml`, `metrics.csv`, and self-contained `best.pt`/`last.pt` checkpoints. Evaluation adds `evaluation.json`, `per_class.csv`, `predictions.json`, `errors.csv`, and annotated visualizations. Prediction JSON preserves floating-point boxes, ordered class names, and manifest identity.

## Models and scope

The registry includes Faster R-CNN MobileNet V3 Large 320 FPN (default), Faster R-CNN ResNet-50 FPN, and SSDLite 320 MobileNet V3 Large. `weights: none` is fully offline; `weights: imagenet1k_v1` explicitly requires a cached backbone or network access.

The reference configuration has **no published full-VOC score**. Repository tests and examples use synthetic or bounded data and do not claim a comparable full-dataset result. The reserved VOC 2007 test split is for final evaluation, not model selection.

## Repository map

- `configs/`: learning, reference, and comparison recipes.
- `src/object_detector/`: typed data, model, training, evaluation, and inference modules.
- `scripts/`: download, preview, and metric plotting tools.
- `examples/`: five progressive local examples.
- `docs/`: tutorials, concepts, guides, and reference.
- `tests/`: offline unit, integration, packaging, and acceptance coverage.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [MIT License](LICENSE).
