# Tutorial 01: Choose a Training Environment

[简体中文](01-environment.zh-CN.md) | [Tutorial index](README.md)

This tutorial recommends training on Kaggle. Kaggle already provides Python,
PyTorch, and an NVIDIA GPU, so you can focus on the data, model, and result
instead of configuring local CUDA and drivers first.

## Recommended: a Kaggle GPU

The project includes a runner you can submit directly. Install the Kaggle CLI
locally:

```bash
uv tool install kaggle
kaggle auth login
```

Then follow the [Kaggle training guide](../guides/kaggle.md) to change the job
owner and submit it. On the web page you need:

- A T4 or newer NVIDIA GPU. Do not use P100; the current PyTorch build does not
  support its `sm_60` compute capability.
- Internet enabled for official VOC 2007 and ImageNet backbone downloads.
- About 60 minutes of runtime.

If Kaggle displays T4 x2, the project still uses only `cuda:0`. An idle second
card does not affect training.

## Optional: inspect the project locally

Local use requires Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/). From
the repository root:

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect --help
uv run detect list-models
```

The version should be `0.1.0`, and help should include `prepare-data`,
`inspect-data`, `train`, `evaluate`, and `predict`. These commands do not start
training or download weights.

Inspect the Kaggle reference configuration:

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

`show-config` only displays the resolved values and their sources. It does not
construct a model. The reference configuration uses an `imagenet1k_v1`
backbone; the Kaggle runner downloads that weight in its networked environment.

## Optional: check local devices

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available()); print('cpu', True)"
uv run python -c "from object_detector.preflight import resolve_device; print(resolve_device('auto'))"
```

`device: auto` tries CUDA, Apple MPS, then CPU. A machine without CUDA is not a
problem; continue on Kaggle. CPU is suitable for examples and a dry run, not a
complete 26-epoch VOC training run.

## Optional: perform one CPU dry run

After Chapter 02 prepares local data, you can run:

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

It reads one batch, performs forward and backward passes plus one parameter
update, then prints `dry-run OK`. It saves no checkpoint and does not mean the
model has completed training.

## Common questions

- `kaggle` is not found: rerun `uv tool install kaggle` and ensure the uv tool
  directory is on PATH.
- The Kaggle API rejects authentication: run `kaggle auth login --force`.
- The Kaggle page has no GPU option: complete platform account verification and
  check your GPU quota.
- Local CUDA is unavailable: use Kaggle; you do not need to reinstall the whole
  local environment to learn this project.
- `uv sync --locked` fails: confirm Python is 3.10-3.12.
- P100 reports `no kernel image`: select a T4 or newer GPU.

Continue to [VOC data and bounding boxes](02-data-and-boxes.md).
