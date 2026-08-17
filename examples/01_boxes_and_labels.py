from __future__ import annotations

import argparse

import torch


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Inspect xyxy boxes and integer class labels")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    boxes = torch.tensor([[2.0, 3.0, 18.0, 15.0], [20.0, 4.0, 30.0, 22.0]])
    labels = torch.tensor([1, 3], dtype=torch.int64)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    print(f"boxes={boxes.tolist()}")
    print(f"labels={labels.tolist()} areas={areas.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
