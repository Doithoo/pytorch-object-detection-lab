# 00 - Detection basics

Each image is a `[3,H,W]` float tensor. A target is a dictionary whose `boxes` are float `[N,4]` xyxy coordinates and whose `labels` are int64 class IDs. Images and object counts may differ inside one batch, so detection uses lists instead of stacking.

Run: `uv run python examples/02_detection_batch.py`

Expected: two different image shapes and target counts `1` and `2` are printed.
