# Learning path

The sequence is `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`. Preparation creates a stable manifest identity; every later checkpoint and report carries that identity. Dry run proves one update before a longer bounded run.

Run: `uv run detect --version`

Expected: `0.1.0` is printed without loading a dataset or model.
