#!/usr/bin/env python3
"""Check native candidate scoring against independent PyTorch MLP inference."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = json.loads((args.model / "reranker.json").read_text())
    cases = json.loads((args.model / "verification.json").read_text())
    inputs = "".join(json.dumps({"points": case["points"]}) + "\n" for case in cases)
    binary = ROOT / "target/release/examples/retrieval_cli"
    def run(extra):
        result = subprocess.run([str(binary), "--model", str(args.model), *extra], input=inputs, capture_output=True, text=True, check=True)
        return [json.loads(line) for line in result.stdout.splitlines()]
    features = run(["--features", "--candidates", str(weights["shortlist"])])
    predictions = run([])
    exported = []
    for case, candidates, native in zip(cases, features, predictions):
        raw = torch.tensor([c["features"] for c in candidates["candidates"]])
        values = ((raw - torch.tensor(weights["mean"])) / torch.tensor(weights["scale"])).clamp(-12, 12)
        for i, layer in enumerate(weights["layers"]):
            values = F.linear(values, torch.tensor(layer["weights"]), torch.tensor(layer["bias"]))
            if i + 1 != len(weights["layers"]):
                values = values.relu()
        costs = (raw[:, 1] + weights["residual_weight"] * values[:, 0]).numpy()
        indices = sorted(range(len(costs)), key=lambda i: (float(costs[i]), candidates["candidates"][i]["word"]))[:5]
        expected = [{"word": candidates["candidates"][i]["word"], "rating": float(costs[i])} for i in indices]
        actual = native["predictions"]
        if [p["word"] for p in actual] != [p["word"] for p in expected]:
            raise ValueError(f"native ranking differs for {case['word']}")
        error = np.max(np.abs([a["rating"] - e["rating"] for a, e in zip(actual, expected)]))
        if error > 1e-5:
            raise ValueError(f"native score error {error}")
        exported.append({"word": case["word"], "points": case["points"], "predictions": expected})
        print(f"{case['word']}: maximum native ranker error {error:.8f}")
    args.output.write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    main()
