import torch

from object_detector.models.registry import build_model


def test_default_detector_completes_one_optimization_step() -> None:
    model = build_model("fasterrcnn_mobilenet_v3_large_320_fpn", 21, "none", {})
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    image = torch.rand(3, 64, 64)
    target = {
        "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    losses = model([image], [target])
    total = sum(losses.values())
    assert losses
    assert torch.isfinite(total)
    total.backward()
    optimizer.step()
