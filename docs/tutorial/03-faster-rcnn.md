# 03 - Faster R-CNN

The backbone and FPN produce multi-scale features. The region proposal network suggests class-agnostic regions; ROI heads classify them and regress final boxes. Torchvision owns these internals, while this repository owns class count, weight policy, data contracts, and artifacts.

Run: `uv run python examples/03_detector_losses.py`

Expected: named classification and box-regression losses plus their total are printed.
