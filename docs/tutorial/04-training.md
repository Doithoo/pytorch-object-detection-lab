# Tutorial 04: Train Faster R-CNN on Kaggle

[简体中文](04-training.zh-CN.md) | [Tutorial index](README.md)

This chapter combines the earlier data and model ideas in one complete run.
Use a Kaggle T4 for the recommended path; full VOC training is not a good use
of an ordinary CPU.

## What this run uses

The reference configuration is
[`../../configs/reference_fasterrcnn.yaml`](../../configs/reference_fasterrcnn.yaml):

- Faster R-CNN MobileNet V3 Large 320 FPN.
- ImageNet1K V1 backbone weights.
- Official VOC 2007 train / valid / test splits.
- 26 epochs with SGD and a step learning-rate schedule.
- Validation `map_50_95` to select `best.pt`.

The Kaggle runner changes the device to CUDA, enables AMP, uses two data
workers, and places paths under `/kaggle/working`. The model structure and
optimization settings remain the same.

## Submit training

Complete sign-in and account-name setup in the [Kaggle guide](../guides/kaggle.md),
then run:

```bash
kaggle kernels push -p docs/recorded-run/kaggle
```

Confirm a T4 or newer GPU and enabled Internet on the web page. The runner
automatically downloads and prepares data and performs a one-batch dry run; you
do not run the local preparation commands separately on Kaggle.

## What the dry run checks

Before the first epoch, the runner reads one real VOC batch and completes one
parameter update. Diagnostics include:

```text
image_shapes=((3, H1, W1), (3, H2, W2))
target_counts=(N1, N2)
loss_total=<finite value>
loss_classifier=<finite value>
loss_box_reg=<finite value>
loss_objectness=<finite value>
loss_rpn_box_reg=<finite value>
dry-run OK
```

`dry-run OK` means data reached the model, losses backpropagated, and the
optimizer updated parameters. It is not a training result; training begins with
the epoch logs that follow.

## Read the epoch log

Each epoch has two phases:

1. train: compute losses, backpropagate, and update the model.
2. valid: keep parameters fixed and compute validation mAP and recall.

The runner prints a heartbeat every 60 seconds during long training or
evaluation phases. If heartbeats continue, do not stop or resubmit. The completed
Kaggle run spent 3,025.660 seconds, about 50 minutes, training.

Each epoch is also written to `metrics.csv`. Start with:

| Column | Meaning |
|---|---|
| `epoch` | Completed epoch |
| `loss_total` | Sum of the four training losses |
| `valid_map_50_95` | Main validation metric for checkpoint selection |
| `valid_map_50` | Validation AP at IoU 0.5 |
| `learning_rate` | Current learning rate |

Loss is the optimization objective and mAP measures validation detections. They
do not have to rise or fall together.

## Why both best.pt and last.pt are saved

- `best.pt`: the epoch with the highest validation `map_50_95` so far, used for
  final evaluation and prediction.
- `last.pt`: the latest completed epoch, including optimizer, scheduler, and
  random state needed to resume.

The published run reached its best validation `map_50_95 = 0.313245` at epoch
18, then completed the planned 26 epochs. Final test evaluation used epoch 18
`best.pt`, not epoch 26 `last.pt`.

## What appears after training

`artifacts/reference-fasterrcnn/` contains:

```text
config.yaml
run.yaml
metrics.csv
best.pt
last.pt
evaluation/
```

`config.yaml` is the actual run configuration, `run.yaml` records device, data,
and versions, and `metrics.csv` is the complete history. Do not download only
the checkpoint; the small text files explain how it was produced.

## Optional: one small local check

With local VOC data but no GPU, check one batch and one update:

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

`learning_minimal.yaml` uses random weights and a few samples. It checks the
code path and is not a training result.

## Optional: complete training on a local GPU

If you have a compatible CUDA GPU, full VOC data, and network access for the
backbone weight, run:

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml --device cuda
```

A new run needs an output directory that does not already exist. To continue an
interrupted run, use `last.pt`:

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml --resume artifacts/reference-fasterrcnn/last.pt --device cuda
```

The model, data, and settings that define the training must match the
checkpoint. See the [checkpoint reference](../reference/checkpoint-schema.md)
for the complete field list.

Continue to [evaluation and prediction](05-evaluation-and-inference.md) to
analyze what the real Kaggle model gets right and wrong.
