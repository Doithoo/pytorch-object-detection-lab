# 01 - Environment

uv installs the locked runtime and development dependencies for Python 3.10-3.12. The learning configuration uses random detector initialization and stays offline. The ImageNet backbone policy is explicit and may need network access when the cache is empty.

Run: `uv run detect show-config --config configs/learning_minimal.yaml`

Expected: resolved YAML shows `weights: none`, the default detector, and bounded sample limits.
