# Recorded Full-VOC Run

[Simplified Chinese](README.zh-CN.md) | [Reference configuration](../../configs/reference_fasterrcnn.yaml) | [Kaggle guide](../guides/kaggle.md)

This directory records one completed, evidence-backed Pascal VOC 2007 run. It
is a reproducible project result, not a claim that this recipe is a universal
torchvision benchmark or the best detector for VOC.

## Result

| Field | Recorded value |
|---|---:|
| Model | Faster R-CNN MobileNet V3 Large 320 FPN |
| Backbone weights | ImageNet1K V1 |
| Completed epochs | 26 |
| Best validation epoch | 18 |
| Best validation `map_50_95` | 0.313245 |
| Test images | 4,952 |
| Test targets / predictions | 12,032 / 26,353 |
| Test `map_50_95` | **0.322312** |
| Test `map_50` / `map_75` | 0.609917 / 0.302681 |
| Test `mar_1` / `mar_10` / `mar_100` | 0.338981 / 0.413547 / 0.415008 |
| Training / test evaluation time | 3,025.660 s / 74.893 s |
| Kaggle notebook total | 3,223.9 s |

AP/AR comes from torchmetrics 1.9.0 with pycocotools 2.0.11. It is the
COCO-style IoU sweep implemented by this project, not the historical VOC 2007
11-point evaluator. Values are unitless fractions.

![Test image 000001 with targets and predictions](evaluation/visualizations/summary.png)

Green boxes are ordinary targets, dashed orange boxes are difficult targets,
and blue boxes are predictions. See the recorded
[false-positive example](evaluation/visualizations/false_positive-01-009040.png)
and [missed-object example](evaluation/visualizations/missed-01-006500.png) for
failure cases rather than treating the summary image as representative of the
whole test split.

## Run identity

| Field | Recorded value |
|---|---|
| Completed | 2026-08-19 |
| Kaggle kernel | `yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7` |
| Kernel URL | <https://www.kaggle.com/code/yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7> |
| Device | `cuda:0`, Tesla T4; two GPUs were allocated, one was used |
| Python | 3.12.13 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| Seed | 42 |
| Dataset identity | `b9bdc2604322c07f26c9a0135a063c7702b0dfb261171401076cf6733cfdb5b7` |
| Train / valid / test images | 2,501 / 2,510 / 4,952 |
| Embedded source archive | 157,993 bytes; SHA-256 `2186866a9b4b582e2c2c38128a178bd958ffaa5b0dcafbf6d4c55e4f39aca628` |
| Evaluated checkpoint | SHA-256 `826e2bb38b985945fbfbaf59587e06ecb9fc70501c5ce80f6d1e357b59b0826a` |

`run.yaml` records `git_revision: null` because the Kaggle runner embedded the
current uncommitted project snapshot rather than a Git commit. The exact
157,993-byte source archive remains embedded in
[`kaggle/run_kaggle.py`](kaggle/run_kaggle.py), and its digest is recorded
above. This preserves the executed source without pretending a commit existed.

The official split hashes are preserved in [`run.yaml`](run.yaml). The resolved
[`config.yaml`](config.yaml) records the Kaggle-specific overrides: CUDA, AMP,
two data workers, and `/kaggle/working` paths. Model selection used validation
`map_50_95`; the reserved test split was evaluated only after epoch 26 finished.

## Preserved evidence

- [`metrics.csv`](metrics.csv): all 26 training and validation rows.
- [`kaggle-run-summary.json`](kaggle-run-summary.json): split counts, elapsed
  times, and unrounded test metrics.
- [`evaluation/evaluation.json`](evaluation/evaluation.json): rounded test
  metrics, thresholds, backend versions, dataset identity, and checkpoint hash.
- [`evaluation/per_class.csv`](evaluation/per_class.csv): all 20 VOC classes.
- [`evaluation/errors.csv`](evaluation/errors.csv): recorded ignored,
  localization, false-positive, and missed rows.
- [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) and
  [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json): exact submitted
  runner and Kaggle machine settings.

The 145 MB checkpoint, downloaded VOC data, and 5.6 MB prediction listing are
not committed. The checkpoint hash above binds this report to the evaluated
file. The three retained images are selected evidence, not a curated accuracy
claim.

## Reproduce or download

The exact runner is self-contained and needs Kaggle internet access for the
official VOC archives and ImageNet backbone weight. Change the metadata `id` to
your Kaggle account before pushing a separate copy:

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7
kaggle kernels output yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7 --file-pattern 'artifacts/.*' -p kaggle-output
```

Request a T4 or newer GPU. The current Kaggle PyTorch build does not include
P100 `sm_60` kernels, so a P100 fails before training. The runner deliberately
uses one GPU and does not claim multi-GPU scaling. See the [Kaggle
guide](../guides/kaggle.md) for the short operational workflow.
