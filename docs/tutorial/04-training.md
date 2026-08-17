# 04 - Training

Training resolves configuration, validates dataset/model identity, seeds random generators, and writes atomic checkpoints. `--dry-run` performs exactly one update and prints shapes, target counts, and losses. Resume may change epochs, workers, device, output directory, or run name, but not training semantics.

Run: `uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu`

Expected: diagnostics end with `dry-run OK`; no run directory or checkpoint is written.
