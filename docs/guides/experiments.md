# Compare Two Training Runs

[简体中文](experiments.zh-CN.md) | [Kaggle training record](../recorded-run/README.md)

After your first Kaggle run, change one setting at a time to understand its
effect. The project provides one Faster R-CNN MobileNet result as a known
starting point, not a leaderboard.

## Start with one clear question

Useful questions include:

- How does a ResNet-50 backbone differ from MobileNet with the same training setup?
- With the model fixed, how does learning rate affect validation metrics?
- How do Faster R-CNN and SSDLite compare with the same data and epochs?

Do not change model, weights, learning rate, and epoch count together. The
result would be difficult to explain.

## Keep these values fixed

- Train / validation / test splits.
- Random seed.
- Epoch count and sample limits.
- Optimizer, scheduler, and data augmentation.
- Validation and test metrics.

Give every run a different `run_name` and output directory. Keep its
`config.yaml` and `metrics.csv`.

## Inspect configurations first

For example, change only the model:

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set run_name experiment-a
uv run detect show-config --config configs/learning_minimal.yaml --set run_name experiment-b --set model.name ssdlite320_mobilenet_v3_large
```

You can perform local dry runs to confirm that both models read data and update
once:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name experiment-a --dry-run --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name experiment-b --set model.name ssdlite320_mobilenet_v3_large --dry-run --device cpu
```

A dry run saves no model and compares no accuracy. For a real comparison, run
both configurations on a Kaggle GPU with the same data and training budget.

## Compare validation metrics

After both runs finish:

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

The command selects the best validation row for each run and lists important
configuration differences. Alongside the ranking, inspect:

- Trends across `metrics.csv`.
- Per-class AP and recall.
- False-positive and missed-object images.
- Whether runtime and memory fit your use case.

## Look at test only at the end

Use validation to choose the configuration and checkpoint. After all choices
are fixed, evaluate test once for the selected setting. Do not repeatedly tune
settings from test results.

A run with sample limits or very few epochs only describes that small trial.
The project's only published complete VOC result remains the
[Kaggle v7 training run](../recorded-run/README.md).
