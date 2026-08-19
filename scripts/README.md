# Utility Scripts

[Simplified Chinese](README.zh-CN.md) | [Workflow tutorial](../docs/tutorial/README.md)

Run scripts from the repository root through `uv run python`. Each script has one clear job; none silently trains or evaluates a detector.

## Script index

| File | Role and prerequisite | Network behavior | Expected artifact or output |
|---|---|---|---|
| `download_data.py` | Download and safely extract the two official VOC 2007 archives; requires write access to the data directory | Uses the official VOC HTTP URLs when a verified archive is not already cached; validates published MD5 checksums | Archives under `<data-dir>/archives` and extracted `VOCdevkit/VOC2007`; prints archive paths |
| `preview_dataset.py` | Render prepared manifest samples and boxes; requires manifests and local source images | Offline | A PNG, defaulting to `artifacts/dataset_preview.png`; prints the output path |
| `plot_metrics.py` | Plot every `loss*` column from a training `metrics.csv`; requires the development extra for matplotlib | Offline | A caller-selected PNG; prints the output path and rejects empty CSVs or files without loss columns |
| `generate_doc_assets.py` | Regenerate deterministic synthetic teaching diagrams through project rendering code | Offline; uses generated tensors and the default bundled font | `detection-target-anatomy.png` and `detection-error-analysis.png` in the selected directory |
| `__init__.py` | Marks `scripts` as an importable package for tests and reusable helpers | No network | No command and no artifact |

## Workflow commands

Download is the only script whose normal path can access the network:

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

It verifies each archive before extraction and rejects unsafe or conflicting files. Successful download is only the source-data stage. Continue with `detect prepare-data`, then inspect the prepared data:

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split train --limit 4 --output artifacts/dataset_preview.png
```

After a normal training run, plot its loss columns:

```bash
uv run python scripts/plot_metrics.py --metrics artifacts/run/metrics.csv --output artifacts/run/losses.png
```

Regenerate documentation images only as an explicit maintenance action:

```bash
uv run python scripts/generate_doc_assets.py --output-dir docs/assets
```

## Scope

The download and preview scripts check source files and make annotations visible; they do not measure model quality. The metrics plot displays recorded loss columns but does not recompute metrics or show convergence. Documentation assets are synthetic diagrams. None of these outputs is a full VOC result. The completed result is documented in the [Kaggle training record](../docs/recorded-run/README.md).
