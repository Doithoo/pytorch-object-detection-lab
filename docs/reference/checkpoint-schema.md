# Checkpoint schema

Schema version `1` contains resolved config, model name/params, explicit weight policy, ordered class names, preprocessing contract, manifest identity and split hashes, model/optimizer/scheduler states, epoch, best metric, metric history, and run metadata. Evaluation and prediction rebuild the model with `weights=none`; no YAML or download is needed.

Resume requires matching model name, classes, preprocessing, manifest identity, and semantic configuration. Allowed operational overrides are `train.epochs`, `data.num_workers`, `device`, `output_dir`, and `run_name`. Epoch must extend the saved run. Other changes require a new run rather than resume.
