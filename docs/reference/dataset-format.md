# Dataset format

Preparation expects the official `VOCdevkit/VOC2007` directory and train, val, and test split files. It writes CSV manifests and metadata containing split hashes, ordered classes, source root, coordinate convention, and a combined identity.

VOC boxes are one-based inclusive. Conversion to zero-based continuous xyxy is `(xmin-1, ymin-1, xmax, ymax)`, where the maximum corner is the exclusive pixel boundary. Training removes difficult objects. Validation/test preserve them as `iscrowd=1`; metrics and error analysis ignore their matches and ordinary target counts exclude them. Empty targets use tensors shaped `[0,4]` and `[0]`.
