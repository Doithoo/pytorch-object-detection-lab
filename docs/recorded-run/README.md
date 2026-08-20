# Kaggle VOC 2007 Training Record

[简体中文](README.zh-CN.md) | [Kaggle training guide](../guides/kaggle.md) | [Reference configuration](../../configs/reference_fasterrcnn.yaml)

This is the complete training result published by the project. The job
finished successfully on a Kaggle Tesla T4 on 2026-08-19. It trained for 26
epochs and evaluated official VOC 2007 test data with the epoch 18 model chosen
on validation.

## Result

| Item | Value |
|---|---:|
| Model | Faster R-CNN MobileNet V3 Large 320 FPN |
| Backbone weights | ImageNet1K V1 |
| Completed epochs | 26 |
| Best validation epoch | 18 |
| Best validation `map_50_95` | 0.313245 |
| Test images | 4,952 |
| Test ordinary targets / predictions | 12,032 / 26,353 |
| Test `map_50_95` | **0.322312** |
| Test `map_50` / `map_75` | 0.609917 / 0.302681 |
| Test `mar_1` / `mar_10` / `mar_100` | 0.338981 / 0.413547 / 0.415008 |
| Training / test evaluation time | 3,025.660 s / 74.893 s |
| Complete Kaggle job | 3,223.9 s |

AP/AR was computed with torchmetrics 1.9.0 and pycocotools 2.0.11. This
`map_50_95` is the COCO-style average over several IoU thresholds, not the
historical VOC 2007 11-point calculation. Values are unitless fractions from 0
to 1.

## See what the model did

![Targets and predictions for test image 000001](evaluation/visualizations/summary.png)

Green boxes are ordinary targets, dashed orange boxes are difficult targets,
and blue boxes are model predictions. Continue with two real failure cases:

- [False-positive example](evaluation/visualizations/false_positive-01-009040.png)
- [Missed-object example](evaluation/visualizations/missed-01-006500.png)

One image cannot represent all 4,952 test images. It helps connect aggregate
metrics and CSV rows to a concrete image.

## Files you can inspect directly

| File | Contents |
|---|---|
| [`metrics.csv`](metrics.csv) | Training losses and validation metrics for all 26 epochs |
| [`config.yaml`](config.yaml) | Complete configuration actually used on Kaggle |
| [`run.yaml`](run.yaml) | Data splits, device, versions, seed, and runtime |
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | Epochs, times, split counts, and unrounded test metrics |
| [`evaluation/evaluation.json`](evaluation/evaluation.json) | Test summary, thresholds, and backend versions |
| [`evaluation/per_class.csv`](evaluation/per_class.csv) | Metrics for all 20 VOC classes |
| [`evaluation/errors.csv`](evaluation/errors.csv) | Ignored, localization, false-positive, and missed rows |

Start with `epoch`, `loss_total`, and `valid_map_50_95` in `metrics.csv`, then
open `evaluation.json` and the images. The
[evaluation tutorial](../tutorial/05-evaluation-and-inference.md) explains how
to read them together.

## How the run was made

- Page: <https://www.kaggle.com/code/yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7>
- Device: `cuda:0`, Tesla T4. Kaggle allocated two; the project used one.
- Data: official VOC 2007 with 2,501 / 2,510 / 4,952 train / valid / test images.
- Random seed: 42.
- Training: 26 epochs, CUDA AMP, and two data workers.
- Selection: validation `map_50_95` after every epoch saved the best model.
- Test: one evaluation with `best.pt` after all training finished.

The suffix in the page URL is part of the original Kaggle submission address.
It is not a model, tutorial, or project version, and new runs do not need to
use it.

The submitted files are [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) and
[`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json). The runner embeds
the source, so it needs neither an attached Dataset nor `kagglehub`.

## Run it yourself

Change the metadata `id` to your account, then:

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

Request a T4 or newer GPU and enable Internet. The current Kaggle PyTorch build
does not support P100 `sm_60`, so P100 fails when training begins. See the
[Kaggle guide](../guides/kaggle.md) for complete steps and known failures.

## Reproducibility details

These fields identify the exact run and can be skipped on a first read:

| Item | Recorded value |
|---|---|
| Python | 3.12.13 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| Dataset identity | `b9bdc2604322c07f26c9a0135a063c7702b0dfb261171401076cf6733cfdb5b7` |
| Embedded source size / SHA-256 | 157,993 bytes / `2186866a9b4b582e2c2c38128a178bd958ffaa5b0dcafbf6d4c55e4f39aca628` |
| Evaluated checkpoint SHA-256 | `826e2bb38b985945fbfbaf59587e06ecb9fc70501c5ce80f6d1e357b59b0826a` |

`run.yaml` has `git_revision: null` because the runner saved the source snapshot
that existed at the time rather than a Git commit. The exact source remains
embedded in the runner and is identified by the digest above.

The repository does not commit the 145 MB checkpoint, downloaded VOC data, or
the 5.6 MB full prediction listing. Download `best.pt` from your own Kaggle job
when you need the model. The retained configuration, metrics, error CSV, and
three images make this public result inspectable; they do not claim that this
model is the best detector on VOC.
