# Release Readiness Improvements Design

## Goal

Move the repository from a sound implementation skeleton to a trustworthy,
publishable learning project without inventing benchmark results. The first
stage fixes full-dataset reliability and tightens public contracts before
adding more models or datasets.

## Reliability

Evaluation must not retain every decoded image. Metrics, prediction records,
and error records may accumulate, but visualization keeps only source image
IDs and reloads the small ranked set after metrics finish. Evaluation uses a
DataLoader so inference batch size and workers are explicit checkpoint config
values rather than hidden single-image behavior.

Configuration rejects non-finite numeric values, unsupported optimizer and
scheduler names, empty identifiers, and invalid runtime thresholds before any
model or dataset work. Checkpoint preprocessing metadata is required to match
the complete schema. Resume requires the requested final epoch to be greater
than the saved epoch and refuses to publish into an unrelated nonempty run
directory.

## Documentation And Packaging

Documentation tests execute or parse every documented local command form,
including `uv run detect` and `uv run python`. Missing example files therefore
fail CI. The Faster R-CNN contract example is a real offline program.

The distribution is explicitly a Git-clone-first learning repository. Package
metadata and README say so; wheel installation exposes the reusable package and
CLI, while repository-only scripts, examples, and tutorials are not presented
as wheel contents. The sdist includes those learning resources for source
distribution users.

## Reference Evidence

The repository adds a recorded-run schema and instructions but continues to
state that no full VOC score exists until a real run is completed. Metrics,
images, checkpoint hashes, and hardware/runtime metadata must come from that
run and must never be synthesized for publication.

## Verification

Each behavioral change starts with a focused failing test. Completion requires
the full offline test suite, Ruff, formatting, mypy, lock validation, package
build, Twine checks, and inspection of source-distribution contents.
