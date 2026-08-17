# 02 - VOC data and boxes

VOC XML stores one-based inclusive corners. Parsing converts `(xmin,ymin,xmax,ymax)` to zero-based continuous xyxy by subtracting one from the minimum coordinates and retaining the maximum coordinates. Difficult objects remain in validation/test as `iscrowd=1` but are excluded from training targets and ordinary counts.

Run: `uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --limit 4`

Expected: `artifacts/dataset_preview.png` shows ordinary and dashed difficult boxes.
