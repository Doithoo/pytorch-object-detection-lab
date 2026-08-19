# Security Policy

## Supported Version

Security fixes target the current `main` branch until versioned releases are
published. Release support will be documented here when the policy changes.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
[private security advisory form](https://github.com/Yashowhoo/pytorch-object-detection-lab/security/advisories/new)
and include the affected revision, reproduction, impact, and any suggested
mitigation. Remove private datasets, credentials, and local paths.

Dataset archives and pretrained weights are external inputs. The VOC downloader
verifies published checksums and rejects unsafe tar members; checkpoints load
with PyTorch's tensor-only mode. Report any path that bypasses those boundaries.
