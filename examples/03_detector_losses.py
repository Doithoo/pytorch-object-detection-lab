from __future__ import annotations

import argparse

import torch
from torch import nn


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        return {"loss_classifier": self.scale.square(), "loss_box_reg": self.scale.abs() * 0.5}


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Inspect a detector training loss dictionary")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    model = TinyDetector().train()
    losses = model([torch.zeros((3, 16, 16))], [{"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]])}])
    print({name: float(value.detach()) for name, value in losses.items()})
    print(f"loss_total={float(sum(losses.values()).detach())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
