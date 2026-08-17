# 05 - Evaluation and inference

Evaluation reconstructs the model from a checkpoint, verifies manifest identity before inference, computes COCO-style AP/AR, ranks deterministic errors, and renders evidence. Prediction also reconstructs with `weights=none`, needs no YAML, and retains all JSON detections before applying the visualization display limit.

Run: `uv run detect predict --checkpoint artifacts/run/best.pt --image image.jpg --output-dir artifacts/prediction --device cpu`

Expected: `image.json` and `image.png` are written, or a concise checkpoint/image error is returned with exit code 2.
