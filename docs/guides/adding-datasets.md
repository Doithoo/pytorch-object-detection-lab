# Add an Internal Dataset Provider

[Simplified Chinese](adding-datasets.zh-CN.md) | [Dataset contract](../reference/dataset-format.md)

This maintainer guide applies when the source cannot satisfy the supported VOC-shaped layout. Version 0.1 registers only `voc2007`; `data.name` rejects every other value. There is no stable external provider or plugin API, so this work is an internal code change with a compatibility obligation.

## Preserve the boundary

Keep preparation separate from runtime loading:

```text
source files -> full validation -> staged manifests + metadata -> atomic publication
prepared rows -> Dataset -> (float RGB image, target) -> list-based collate
```

A provider must publish fixed train, valid, and test membership, reject duplicates and cross-split overlap, and derive identity from source content, ordered classes, coordinate rules, and split rows. Manifests reference source paths; they do not copy source data. A failed preparation must leave the previous manifest directory intact.

## Implementation ownership

1. Add format parsing and validation under `src/object_detector/data/`. Keep source-coordinate conversion in the parser.
2. Add a preparation function that writes the schemas documented in [dataset format](../reference/dataset-format.md) through staging and atomic replacement.
3. Add a manifest-backed `Dataset` that returns the canonical target below. Add selection in configuration and orchestration only after defining a real internal registry or an explicit second-provider dispatch.
4. Update preflight, checkpoint identity, evaluation, examples, and bilingual documentation together.

The canonical item is `image: float32 Tensor[3,H,W]` in RGB `[0,1]` plus:

| Field | Contract |
|---|---|
| `boxes` | `float32 [N,4]`, zero-based continuous `xyxy` |
| `labels` | `int64 [N]`; 0 is background and object labels start at 1 |
| `image_id` | `int64 [1]`, stable for the source ID |
| `area` | `float32 [N]`, `(xmax-xmin)*(ymax-ymin)` |
| `iscrowd` | `int64 [N]` |
| `difficult` | `bool [N]` |

Object-aligned fields must share `N`. Empty targets keep `[0,4]` and `[0]` shapes. Geometry transforms must update the image, boxes, area, and every aligned field together. `detection_collate` must return lists so variable image sizes and object counts remain valid.

## Failure and evidence requirements

Reject missing or corrupt images, malformed annotations, unknown labels, non-finite or degenerate boxes, filename or dimension disagreement, invalid split membership, and metadata/class-count mismatch before training. Decide and document difficult/crowd semantics instead of silently dropping them.

Add focused parser, manifest, dataset, transform, inspection, preflight, and end-to-end tests. Include empty and difficult-only images, stable identity, changed source bytes, atomic failure, collate, and a model dry run. Run:

```bash
uv run pytest tests/test_manifest.py tests/test_dataset.py tests/test_transforms.py tests/test_end_to_end.py -q
```

A provider is not complete merely because one sample loads. It is complete when preparation is reproducible, invalid input fails without partial publication, runtime targets satisfy the detector contract, and the identity reaches run artifacts and checkpoints. For already VOC-shaped sources, use [your own data](using-your-data.md) instead.
