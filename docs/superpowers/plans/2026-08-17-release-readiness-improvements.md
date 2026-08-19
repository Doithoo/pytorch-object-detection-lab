# Release Readiness Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each behavioral change. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix full-dataset reliability and public contract gaps, then make the documentation and distribution honest and testable.

**Architecture:** Keep existing module boundaries. Make evaluation retain lightweight image references instead of image tensors, centralize input validation at configuration/CLI boundaries, and enforce checkpoint/resume invariants before artifact writes. Add executable documentation coverage and source-distribution resources without broadening the runtime dependency set.

**Tech Stack:** Python 3.10-3.12, PyTorch, torchvision, pytest, setuptools, uv, GitHub Actions.

---

### Task 1: Stream Evaluation Evidence

**Files:** `tests/test_evaluation.py`, `src/object_detector/evaluation/evaluate.py`

- [x] Add a failing test proving retained evaluation records do not contain image tensors.
- [x] Replace `_EvaluatedImage` with a lightweight image ID/index record.
- [x] Reload only the summary and ranked evidence samples for rendering.
- [x] Run `uv run pytest tests/test_evaluation.py -v`.

### Task 2: Tighten Configuration And CLI Validation

**Files:** `tests/test_config.py`, `tests/test_cli.py`, `src/object_detector/config.py`, `src/object_detector/cli.py`

- [x] Add failing tests for non-finite numbers, invalid optimizer/scheduler names, empty names, invalid thresholds, and negative display limits.
- [x] Add focused finite-number, nonempty-string, enum, probability, and nonnegative-integer validators.
- [x] Apply validation before command handlers construct models or datasets.
- [x] Run the focused config and CLI tests.

### Task 3: Enforce Checkpoint And Resume Contracts

**Files:** `tests/test_checkpoint.py`, `tests/test_training.py`, `src/object_detector/training/checkpoint.py`, `src/object_detector/training/train.py`

- [x] Add failing tests for missing/partial preprocessing metadata and non-extending resume epochs.
- [x] Require exact preprocessing metadata for schema version 1.
- [x] Reject resume when requested epochs do not extend the checkpoint.
- [x] Protect resume destinations that are unrelated nonempty directories.
- [x] Run checkpoint and training tests.

### Task 4: Make Documentation Commands Executable

**Files:** `tests/test_documentation.py`, `examples/03_model_contract.py`, `examples/README.md`, `examples/README.zh-CN.md`, `docs/concepts/how-faster-rcnn-works*.md`

- [x] Add a failing documentation test that discovers missing paths in `uv run python` commands.
- [x] Add the offline model-contract example and document it.
- [x] Parse `uv run detect` command lines with the real CLI parser.
- [x] Run documentation and example tests.

### Task 5: Align Source Distribution And Publication Metadata

**Files:** `MANIFEST.in`, `pyproject.toml`, `README.md`, `README.zh-CN.md`, `tests/test_packaging.py`, `.github/**`, `docs/recorded-run/**`

- [x] Add failing packaging assertions for publication metadata and sdist resources.
- [x] Include docs, examples, scripts, and configs in the sdist while keeping the wheel focused.
- [x] Add project classifiers/keywords/URLs and standard issue/PR templates.
- [x] Add an explicit recorded-run schema and reproduction guide with no fabricated results.
- [x] Build and inspect both distributions.

### Task 6: Complete Verification

- [x] Run `uv run pytest -W error::DeprecationWarning`.
- [x] Run Ruff lint and format checks plus mypy.
- [x] Run `uv lock --check`.
- [x] Build to a temporary directory and run Twine checks.
- [x] Confirm `git diff --check` and inspect final status.
