# Detection flow

An image and target dictionary enter a variable-length collated batch. In training mode a torchvision detector returns named scalar losses; the trainer validates finiteness, sums them, and updates parameters. In evaluation mode it returns boxes, labels, and scores. Those CPU predictions feed torchmetrics unchanged, deterministic error matching, JSON serialization, and rendering.
