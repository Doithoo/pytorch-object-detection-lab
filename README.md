# PyTorch Object Detection for Beginners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-object-detection-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**简体中文: [README.zh-CN.md](README.zh-CN.md)**

This is a beginner-friendly PyTorch object-detection project. You will use
torchvision Faster R-CNN to learn bounding boxes, VOC data, training,
evaluation, and prediction, then complete a real training run on a free Kaggle
GPU.

You do not need to configure local CUDA or read every reference page first.
Start on Kaggle and return to the tutorials when you want to understand a
concept in more depth.

## Completed Kaggle training

The project has completed a 26-epoch VOC 2007 run on a Kaggle Tesla T4. The
validation score selected epoch 18 as `best.pt`, which was evaluated once on
all 4,952 test images.

| Item | Result |
|---|---:|
| Model | Faster R-CNN MobileNet V3 Large 320 FPN |
| Test `mAP@0.5:0.95` | **0.322312** |
| Test `mAP@0.5` | **0.609917** |
| Training time | 3,025.660 seconds, about 50 minutes |
| Complete Kaggle job | 3,223.9 seconds, about 54 minutes |

![Prediction from the Kaggle-trained model on a VOC 2007 test image](docs/recorded-run/evaluation/visualizations/summary.png)

This is a real prediction saved by the completed Kaggle run, not a teaching diagram.
See the [Kaggle run record](docs/recorded-run/README.md) for full metrics,
per-class results, false positives, and missed objects. It is the only complete
training result published by this repository; small runs and synthetic
examples are used only to explain the code.

## Start training on Kaggle

You need a Kaggle account with GPU access. The supplied runner uploads the
source, downloads official VOC 2007, prepares the data, trains, evaluates, and
saves the outputs. It performs the complete sequence:

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

### 1. Get the project and install the Kaggle CLI

```bash
git clone https://github.com/Doithoo/pytorch-object-detection-lab.git
cd pytorch-object-detection-lab
uv tool install kaggle
kaggle auth login
```

The Kaggle CLI is a submission and download tool, not a project training
dependency. This runner does not require `kagglehub` or an attached Kaggle
Dataset.

### 2. Set your Kaggle username

Open `docs/recorded-run/kaggle/kernel-metadata.json` and replace `yashowhoo` in
the `id` with your Kaggle username. Keep `enable_gpu: true` and
`enable_internet: true`.

### 3. Submit and watch the run

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <your-username>/pytorch-object-detection-lab-voc2007-gpu
```

Open the job on Kaggle and confirm it received a T4 or newer GPU. The page may
show T4 x2, but this is a single-GPU project and uses only one card. That is
expected. The log prints a heartbeat every 60 seconds, and the complete run
takes roughly 50-60 minutes.

### 4. Download the result

After the status becomes `COMPLETE`, download only the training artifacts so
you do not also download the temporary VOC directory:

```bash
kaggle kernels output <your-username>/pytorch-object-detection-lab-voc2007-gpu --file-pattern 'artifacts/.*' -p kaggle-output
```

Start with these files under `kaggle-output`:

- `metrics.csv`: training losses and validation metrics for every epoch.
- `best.pt`: the model selected by validation performance.
- `last.pt`: the final epoch and resume state.
- `evaluation/evaluation.json`: the test-set summary.
- `evaluation/per_class.csv`: results for all 20 VOC classes.
- `evaluation/visualizations/`: real predictions, false positives, and misses.

The [Kaggle training guide](docs/guides/kaggle.md) covers account setup,
monitoring, downloads, and the failures already encountered while producing
the recorded run.

## Recommended learning order

You do not need to read everything at once. Follow these pages while you run
the project:

1. [Read the learning path](docs/tutorial/learning-path.md) for the big picture.
2. [Understand images, labels, and boxes](docs/tutorial/00-basics.md).
3. [Meet the VOC dataset](docs/tutorial/02-data-and-boxes.md).
4. [Understand Faster R-CNN](docs/tutorial/03-faster-rcnn.md).
5. [Train on Kaggle](docs/tutorial/04-training.md).
6. [Read evaluation and predictions](docs/tutorial/05-evaluation-and-inference.md).

See the [documentation home](docs/README.md) for every guide and reference.

## Optional: check the project locally

To inspect the environment and commands before submitting to Kaggle, use
Python 3.10-3.12 and run:

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect list-models
uv run detect show-config --config configs/reference_fasterrcnn.yaml
uv run detect verify-data --data-dir data/raw --manifest-dir data/manifests
```

These commands do not start full training. After preparing VOC locally, you
can also perform one small CPU update:

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

Full local training is intended for readers who already have a compatible GPU.
The command and considerations are in the
[optional section of the training tutorial](docs/tutorial/04-training.md).

## What you can learn here

- How VOC XML annotations become torchvision `boxes`, `labels`, and `image_id`.
- Why an object-detection batch is a list of images and a list of targets.
- Why Faster R-CNN returns losses during training and boxes, labels, and scores
  during evaluation.
- How validation selects `best.pt` before one final test-set report.
- How to inspect per-class AP, false positives, misses, and prediction images
  instead of relying on one score.
- How to resume from a checkpoint or run predictions on your own images.

The project includes three model configurations:

| Name | Good for |
|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | Default beginner model and the recorded Kaggle model |
| `fasterrcnn_resnet50_fpn` | A larger Faster R-CNN comparison |
| `ssdlite320_mobilenet_v3_large` | A one-stage detector comparison |

See [choosing a model](docs/guides/using-models.md) and the
[configuration directory](configs/README.md) for details.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
```

Tests use synthetic data and temporary files; they do not download VOC or
pretrained weights. Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing.
The project uses the [MIT License](LICENSE).

<!-- Documentation path: download -> prepare -> inspect -> dry run -> train -> evaluate -> predict | recorded full-VOC score 0.322312 -->
