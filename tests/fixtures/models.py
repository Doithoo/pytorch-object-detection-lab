from __future__ import annotations

import torch
from torch import nn


class FakeDetector(nn.Module):
    def __init__(self, *, nan_loss: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.nan_loss = nan_loss

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None
            classifier = torch.tensor(float("nan"), device=self.scale.device) if self.nan_loss else self.scale.square()
            return {"loss_classifier": classifier, "loss_box_reg": self.scale.abs() * 0.5}
        return [
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]], device=image.device),
                "labels": torch.tensor([1], device=image.device),
                "scores": torch.tensor([0.9], device=image.device),
            }
            for image in images
        ]
