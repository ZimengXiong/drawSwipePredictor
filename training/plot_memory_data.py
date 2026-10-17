#!/usr/bin/env python3
"""Show real recorded gestures alongside synthetic memory-style distortions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from retrieval_model import augment, curve

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "training/data/futo-large/train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/augmentation-examples.png")
    args = parser.parse_args()
    selected = {}
    wanted = {"people", "family", "system", "things"}
    with args.dataset.open() as source:
        for line in source:
            row = json.loads(line)
            if row["word"] in wanted:
                selected.setdefault(row["word"], row)
            if len(selected) == len(wanted):
                break
    if len(selected) != len(wanted):
        raise ValueError("the dataset does not contain the requested example words")
    torch.manual_seed(42)
    fig, axes = plt.subplots(4, 3, figsize=(10, 9), facecolor="#fafaf8")
    titles = ["Human recording", "Mild distortion", "Strong distortion"]
    for r, word in enumerate(sorted(selected)):
        points = curve(selected[word]["points"])
        distorted = augment(torch.tensor(points[None]), force_memory=True)[0].numpy()
        variants = [points, points * 0.6 + distorted * 0.4, distorted]
        for c, values in enumerate(variants):
            axis = axes[r, c]
            axis.set_facecolor("#fafaf8")
            axis.plot(values[:, 0], -values[:, 1], color="#2367ad", linewidth=2)
            axis.scatter(values[0, 0], -values[0, 1], s=24, color="#2b9776", zorder=3)
            axis.scatter(values[-1, 0], -values[-1, 1], s=24, color="#db8143", zorder=3)
            axis.set_aspect("equal", adjustable="datalim")
            axis.axis("off")
            if r == 0:
                axis.set_title(titles[c], fontsize=12, pad=18)
            if c == 0:
                axis.text(-0.1, 0.5, word, transform=axis.transAxes, rotation=90,
                          ha="center", va="center", fontsize=11)
    fig.suptitle("Real motor traces with simulated memory errors", fontsize=17, y=0.98)
    fig.text(0.5, 0.015, "FUTO swipe-1 recordings (MIT). Green: start. Orange: finish. Distorted paths are synthetic.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.95], h_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    print(args.output)


if __name__ == "__main__":
    main()
