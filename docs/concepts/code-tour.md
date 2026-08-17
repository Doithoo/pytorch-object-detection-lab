# Code tour

`config.py` resolves typed settings. `data/` parses VOC, writes manifests, builds samples, and synchronizes transforms. `models/` registers explicit constructors and weight policies. `training/` owns updates, checkpoints, and run artifacts. `evaluation/` owns AP/AR, errors, and evidence. `inference/` restores self-contained checkpoints for local images. `cli.py` is only an argparse adapter.
