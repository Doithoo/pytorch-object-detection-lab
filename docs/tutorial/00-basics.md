# 00 - Detection basics

Classification assigns one label to an image. Detection answers two questions for every object: what is it, and where is it? Each input is a `[3,H,W]` RGB float tensor. A target contains `boxes: float32[N,4]` and `labels: int64[N]`.

A box `[10,20,50,70]` uses `xyxy`: left, top, right, bottom. Its width is `40`, height is `50`, and area is `2000`. This project uses zero-based continuous coordinates with the right and bottom boundaries excluded.

Images and object counts may differ inside one batch, so torchvision detectors receive lists rather than stacked tensors. Empty targets still use tensors shaped `[0,4]` and `[0]`. As an exercise, change one example box, compute its area by hand, and confirm the printed tensor.

Run: `uv run python examples/02_detection_batch.py`

Expected: two different image shapes and target counts `1` and `2` are printed.
