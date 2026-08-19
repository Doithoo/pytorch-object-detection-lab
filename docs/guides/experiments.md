# Run Controlled Experiments

[Simplified Chinese](experiments.zh-CN.md) | [Training tutorial](../tutorial/04-training.md)

This guide is for comparing recipes without losing data provenance or using the test split for iteration. The repository has one [recorded full-VOC run](../recorded-run/README.md), but that single validation-selected result is evidence for its exact recipe, not a leaderboard or a substitute for controlled comparisons.

## Freeze the evidence boundary

Prepare once, record the printed identity, and inspect the source before training:

```bash
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
```

If source image bytes, XML, classes, coordinates, or split membership change, preparation produces a different identity. Do not compare the new run as though it used the old dataset. `compare-runs` rejects different identities.

## State one hypothesis

Begin with `configs/learning_minimal.yaml`, assign a unique run name, and change one semantic field. First inspect typed resolution:

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set run_name baseline
uv run detect show-config --config configs/learning_minimal.yaml --set run_name flip-off --set data.horizontal_flip 0.0
```

Then prove both paths with a dry run before creating artifacts:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name baseline --dry-run --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name flip-off --set data.horizontal_flip 0.0 --dry-run --device cpu
```

A dry run performs one update and prints finite losses, but writes no checkpoint and proves no learning quality. Run the same commands without `--dry-run` only after the contracts pass. A fresh run rejects any existing run directory, so do not reuse names.

## Preserve each run as a unit

A completed training directory contains:

| Artifact | Evidence |
|---|---|
| `config.yaml` | full resolved recipe, not merely the input YAML |
| `run.yaml` | Python/framework/platform/device/seed/git revision, manifest identity, split hashes, ordered classes |
| `metrics.csv` | one row per completed epoch, training losses and validation metrics |
| `best.pt` | last checkpoint that strictly improved validation `map_50_95` |
| `last.pt` | most recently completed epoch plus resume state |

Keep these files together. Resume only to extend the same experiment; it restores optimizer, optional scheduler, metric history, and RNG state, and every descendant inherits the fresh run's `lineage_id`. Every resume checkpoint must record finite configured validation metrics and set `best_metric` to their complete-history maximum. Resuming `last.pt` into a different empty run directory requires its same-lineage sibling `best.pt`, whose strict historical maximum and semantic identity are verified before it is carried forward. Resume directly from `best.pt` only into a new empty run directory or, using the exact original path, when that directory's `last.pt` is missing; an existing in-place `last.pt` must be used instead. Only `train.epochs`, `data.num_workers`, `device`, `output_dir`, and `run_name` may differ, and the requested epoch must exceed the saved epoch. Other changes require a new run.

## Select on validation, report test once

Use `valid_map_50_95` or another recorded validation column to compare compatible runs:

```bash
uv run detect compare-runs artifacts/baseline artifacts/flip-off --metric valid_map_50_95 --output artifacts/flip-comparison.csv
```

The command ranks the best row for each run, displays semantic config differences, and ignores operational `run_name`, `output_dir`, `device`, and `data.num_workers`. Difference values follow the ranked row order and are labeled as `run=value`. A metric name containing `loss` sorts lower first; other metrics sort higher first. It does not decide that one recipe is universally better.

Choose the recipe and checkpoint using validation. Freeze the score/error thresholds, then evaluate the reserved test split once:

```bash
uv run detect evaluate --checkpoint artifacts/baseline/best.pt --split test --output-dir artifacts/baseline/evaluation-test --device cpu
```

If the config has sample limits, the test result is bounded evidence, not a full VOC result. Repeatedly reading test and changing the recipe turns test into validation. Use the [metrics reference](../reference/metrics.md) to interpret artifacts and the [recorded-run gate](../recorded-run/README.md) before publishing any full-VOC claim.
