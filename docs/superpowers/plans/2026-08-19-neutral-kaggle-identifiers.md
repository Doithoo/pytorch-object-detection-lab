# Neutral Kaggle Identifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove internal Kaggle submission numbers from beginner-facing documentation while preserving the real completed-run URL.

**Architecture:** Treat the Kaggle kernel slug as a stable user-facing identifier. Enforce the terminology in publication-page tests, allow the one immutable historical URL, and update English and Chinese documentation together.

**Tech Stack:** Markdown, Kaggle kernel metadata JSON, pytest

---

### Task 1: Add Documentation Guardrails

**Files:**
- Modify: `tests/test_documentation.py`

- [x] Add a test that scans all publication pages for numbered Kaggle labels.
- [x] Allow only the exact recorded external Kaggle URL.
- [x] Add a test that requires the neutral kernel ID and title in metadata.
- [x] Run `uv run pytest tests/test_documentation.py -q` and confirm failure on numbered prose and metadata.

### Task 2: Use a Stable Submission Name

**Files:**
- Modify: `docs/recorded-run/kaggle/kernel-metadata.json`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/guides/kaggle.md`
- Modify: `docs/guides/kaggle.zh-CN.md`

- [x] Change new submissions to `pytorch-object-detection-lab-voc2007-gpu`.
- [x] Change the metadata title to `PyTorch Object Detection Lab VOC2007 GPU`.
- [x] Update every status and output command to use the stable slug.

### Task 3: Remove Internal Revision Language

**Files:**
- Modify: `configs/README.md`
- Modify: `configs/README.zh-CN.md`
- Modify: `docs/guides/experiments.md`
- Modify: `docs/guides/experiments.zh-CN.md`
- Modify: `docs/guides/troubleshooting.md`
- Modify: `docs/guides/troubleshooting.zh-CN.md`
- Modify: `docs/tutorial/*.md`
- Modify: `docs/recorded-run/README.md`
- Modify: `docs/recorded-run/README.zh-CN.md`
- Modify: earlier internal documentation that repeats the label

- [x] Replace numbered labels with "completed Kaggle training run" or the natural Chinese equivalent.
- [x] Describe failed attempts as earlier runners rather than numbered versions.
- [x] Keep the historical URL only in the reproduction section and explain its suffix once.

### Task 4: Verify and Commit

**Files:**
- Test: `tests/test_documentation.py`

- [x] Run the focused documentation tests.
- [x] Search for all remaining numbered Kaggle labels and inspect every match.
- [x] Run Ruff, formatting, mypy, and the full pytest suite.
- [x] Run `git diff --check`, review the final diff, and prepare one documentation commit.
