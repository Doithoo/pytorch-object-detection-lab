from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot detector training metrics")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "epoch" not in rows[0]:
        raise ValueError("metrics CSV must contain at least one row and an epoch column")
    loss_columns = [name for name in rows[0] if name.startswith("loss")]
    if not loss_columns:
        raise ValueError("metrics CSV must contain at least one loss column")
    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    for name in loss_columns:
        axis.plot(epochs, [float(row[name]) for row in rows], marker="o", label=name)
    axis.set(xlabel="epoch", ylabel="loss", title="Training losses")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
