#!/usr/bin/env python3
"""Measure inference latency separately from concurrent training and checks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    buckets = defaultdict(list)
    with args.dataset.open() as source:
        for line in source:
            row = json.loads(line)
            if len(buckets[row["group"]]) < 100:
                buckets[row["group"]].append(row)
    # Interleave groups, warm the model, then repeat a fixed mix three times.
    rows = [row for batch in zip(*[buckets[key] for key in sorted(buckets)]) for row in batch]
    measured = rows * 3
    report = {"timing_only": True, "samples_per_model": len(measured), "warmup": 10,
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "models": {}}
    for name, directory in [("baseline", args.baseline), ("challenger", args.challenger)]:
        with tempfile.TemporaryFile(mode="w+") as inputs, tempfile.TemporaryFile(mode="w+") as output:
            for row in rows[:10] + measured:
                inputs.write(json.dumps({"points": row["points"]}) + "\n")
            inputs.seek(0)
            subprocess.run([str(ROOT / "target/release/examples/retrieval_cli"), "--model", str(directory)], stdin=inputs, stdout=output, check=True)
            output.seek(0)
            times = [json.loads(line)["milliseconds"] for line in output][10:]
        if len(times) != len(measured):
            raise RuntimeError("incomplete timing run")
        report["models"][name] = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                                  "encoder_sha256": hashlib.sha256((directory / "encoder.bin").read_bytes()).hexdigest()}
        if (directory / "reranker.json").exists():
            report["models"][name]["reranker_sha256"] = hashlib.sha256((directory / "reranker.json").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
