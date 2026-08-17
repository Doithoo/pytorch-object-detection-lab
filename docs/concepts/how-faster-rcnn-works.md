# How Faster R-CNN works

The backbone extracts features and FPN exposes several resolutions. The RPN scores anchors and proposes regions. ROI Align samples each region; ROI heads classify foreground/background and regress box corrections. Non-maximum suppression removes duplicate high-overlap predictions.

The interface changes with mode. `model.train()` needs images and targets and returns RPN and ROI loss terms. `model.eval()` needs only images and returns boxes, labels, and scores. Evaluating in training mode is therefore an API error, not merely a statistical difference.

The torchvision image transform owns normalization, resizing, and batching. Checkpoints record this preprocessing contract so inference cannot silently apply a different convention.

Run: `uv run python examples/03_model_contract.py`

Expected: training prints named losses; evaluation prints `boxes`, `labels`, and `scores`.
