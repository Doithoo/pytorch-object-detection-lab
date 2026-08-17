# Adding models

Register a stable name, constructor, supported weight policies, and default parameters in `models/registry.py`. The constructor accepts `num_classes`, weight policy, and model parameters. `none` must avoid downloads. Add registry validation, forward smoke, checkpoint restoration, and offline preflight tests. Do not reimplement detector internals already maintained by torchvision.
