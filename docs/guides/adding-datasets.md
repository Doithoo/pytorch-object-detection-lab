# Adding datasets

Add a parser that validates source annotations and a preparation step that writes deterministic split manifests plus metadata identity. Implement the same `(image, target)` interface and source image IDs as `VocDetectionDataset`. Reuse synchronized transforms and `detection_collate`. Add empty-target, coordinate, difficult/crowd, hash, and preview tests without network access.
