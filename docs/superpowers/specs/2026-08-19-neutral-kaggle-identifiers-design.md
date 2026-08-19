# Neutral Kaggle Identifiers Design

## Goal

Make the documentation understandable without exposing internal Kaggle
submission history. A first-time reader should see one completed training run
and one stable way to submit their own run.

## Decisions

- User-facing prose calls the published result the "completed Kaggle training
  run" or "Kaggle training record." It does not call it `v7`.
- New submissions use the stable kernel slug
  `pytorch-object-detection-lab-voc2007-gpu` in metadata and commands.
- The existing Kaggle page URL remains in the recorded-run reproduction
  section because it identifies the real external result. A nearby sentence
  explains that its `v7` suffix belongs to the historical URL and is not a
  model, tutorial, or project version.
- Troubleshooting describes "earlier runners" and "the current runner"
  instead of recounting numbered development attempts.
- English and Chinese documentation use the same terminology and commands.
- A documentation test rejects numbered Kaggle runner labels outside the one
  allowed historical URL.

## Scope

Update the root READMEs, Kaggle and troubleshooting guides, tutorials,
configuration guides, recorded-run pages, Kaggle metadata, and internal
documentation that still describes the published result as `v7`. Do not
rename the historical external Kaggle page or alter recorded metrics.

## Verification

- Search all tracked text for Kaggle version labels and review every remaining
  match.
- Run documentation tests and the complete project verification suite.
- Confirm local Markdown links and documented commands still pass their
  existing checks.

## Success Criteria

A new reader can submit, monitor, and download a Kaggle run without learning an
internal revision number. The only remaining `v7` text is the immutable URL of
the recorded external run, accompanied by its explanation.
