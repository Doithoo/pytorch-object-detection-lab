# Tutorial 01: Environment, Devices, and Trust Boundaries

[Simplified Chinese](01-environment.zh-CN.md) | [Tutorial index](README.md)

The goal is a repeatable environment whose network and hardware decisions are
visible. You need Python 3.10-3.12, `uv`, and a clone of this repository. VOC is
not required for the checks before the training dry run.

## Install exactly the locked environment

From the repository root:

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect --help
```

`uv sync --locked` installs the resolution already recorded in `uv.lock`; it
fails instead of silently changing that resolution. `--extra dev` installs the
test and documentation tooling used in this lab. The version command prints
`0.1.0` for this checkout, and help lists commands such as `prepare-data`,
`inspect-data`, `train`, `evaluate`, and `predict` without loading a model.

Confirm that Python, PyTorch, and the package resolve from the same environment:

```bash
uv run python -c "import sys, torch, object_detector; print(sys.version); print(torch.__version__); print(object_detector.__file__)"
```

The last path should point at this repository's `src/object_detector`. A path
from another checkout is an environment problem, not a data or detector problem.

## Check CPU, CUDA, and Apple MPS

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available()); print('cpu', True)"
```

The project API resolves `device: auto` in the order CUDA, MPS, CPU. To inspect
the exact result on this machine:

```bash
uv run python -c "from object_detector.preflight import resolve_device; print(resolve_device('auto'))"
```

Start with `--device cpu` when diagnosing the integrated pipeline. Explicit
`cuda` or `mps` training requests are rejected by preflight when unavailable.
MPS training uses full precision; AMP scaling is enabled only for CUDA. This
project is single-device and does not implement distributed training.

After Chapter 02 has prepared data, the production boundary check is:

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

Expected output includes the native image shapes, object counts, named finite
losses, and `dry-run OK`. A dry run performs one optimizer update in memory but
does not publish a run directory or checkpoint.

## Resolve configuration before allocating a model

```bash
uv run detect show-config --config configs/learning_minimal.yaml
```

Expected YAML includes `weights: none`, sample limits of 32 train, 16 valid, and
16 test images, two epochs, `num_workers: 0`, and `device: auto`. It also labels
each leaf source as default or YAML. Use the same command with `--set KEY VALUE`
before an experiment when you need to verify an override.

## The weight policy is a network boundary

`configs/learning_minimal.yaml` sets `model.weights: none`. The registry passes
both detector and backbone weights as `None`, so construction does not request
pretrained weights. This is the reliable offline teaching path.

`model.weights: imagenet1k_v1` is different. Preflight checks the expected local
Torch Hub checkpoint path. If the file is absent, it prints a notice that model
construction requires network access. The repository does not promise that the
network is available or that a cache contains the right file. Decide that policy
before starting; do not infer it from a model name.

Evaluation and prediction rebuild from a self-contained project checkpoint with
`weights=none`, then load `model_state`. They need the local checkpoint and, for
evaluation, matching prepared data; they do not need to fetch backbone weights.

## The dataset download is a separate trust boundary

The model's offline weight policy does not make VOC appear locally. Chapter 02
uses `scripts/download_data.py`, which accesses the official Oxford VOC HTTP
URLs only when a correctly checksummed archive is not already present. It writes
`.part` files during transfer, verifies the published MD5, rejects unsafe tar
members, and then extracts. Network availability itself is outside this
repository's guarantees.

## Common failure boundaries

- `uv sync --locked` reports lock incompatibility: do not remove `--locked` to
  conceal it; check the supported Python version and the committed lockfile.
- `uv run detect` is missing but Python imports work elsewhere: the wrong
  environment or checkout is active.
- CUDA is reported unavailable: check the installed PyTorch build and driver
  outside the lab before changing detector settings.
- MPS fails on an operation: reproduce on CPU with `num_workers: 0` to separate
  a backend issue from a data issue.
- Training prints a pretrained-weight cache notice: the selected policy is not
  an offline guarantee on this machine.
- The dry run fails on missing `dataset.yaml`: environment checks passed, but
  the data preparation boundary has not.

Continue to [Tutorial 02](02-data-and-boxes.md) to download, validate, freeze,
inspect, and preview the data used by later commands.
