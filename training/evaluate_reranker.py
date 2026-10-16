#!/usr/bin/env python3
"""Compare frozen recognizers on fresh donors and held-out synthetic writers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from retrieval_model import augment, curve

ROOT = Path(__file__).resolve().parents[1]


def freeze(args):
    excluded = set()
    for split in ["train", "validation", "test"]:
        with (ROOT / "training/data/futo-large" / f"{split}.jsonl").open() as source:
            excluded.update(json.loads(line)["session"] for line in source)
    donors = defaultdict(list)
    with (args.real_dir / "test.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["session"] not in excluded:
                donors[row["session"]].append(row)
    chooser = np.random.default_rng(8675309)
    selected = []
    for donor in sorted(donors):
        rows = donors[donor]
        for i in chooser.permutation(len(rows))[:10]:
            selected.append(rows[i])
    chooser.shuffle(selected)
    selected = selected[:1000]
    if len(selected) < 300 or len({r["session"] for r in selected}) < 30:
        raise ValueError("need at least 300 fresh recordings from 30 new donors")
    rows = [{"word": r["word"], "points": r["points"], "group": "real", "cluster": r["session"]} for r in selected]
    torch.manual_seed(8675309)
    paths = augment(torch.tensor(np.stack([curve(r["points"]) for r in selected]), device="mps"), force_memory=True).cpu().numpy()
    rows.extend({"word": r["word"], "points": path.tolist(), "group": "distorted_real", "cluster": r["session"]} for r, path in zip(selected, paths))
    with (args.synthetic_dir / "test.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            rows.append({"word": row["word"], "points": row["points"], "group": "synthetic_" + row["difficulty"].lower(), "cluster": str(row["writer_seed"])})
    args.frozen.parent.mkdir(parents=True, exist_ok=True)
    with args.frozen.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {"seed": 8675309, "fresh_donors": len({r["session"] for r in selected}), "real_samples": len(selected),
                "exclusion": "All donor sessions in the previously collected 150k train, 10k validation and 5k test files",
                "selection": "Up to 10 random records per fresh donor, shuffled and capped at 1000; no model scores used",
                "sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
                    "real_test": args.real_dir / "test.jsonl", "synthetic_test": args.synthetic_dir / "test.jsonl"}.items()},
                "limitations": "Recorded swipes use visible keyboards. Distorted and synthetic gestures do not establish human freehand-memory accuracy."}
    args.frozen.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def predict(directory, frozen, count):
    with frozen.open() as source, tempfile.TemporaryFile(mode="w+") as output:
        subprocess.run([str(ROOT / "target/release/examples/retrieval_cli"), "--model", str(directory)], stdin=source, stdout=output, check=True)
        output.seek(0)
        results = [json.loads(line) for line in output]
    if len(results) != count:
        raise RuntimeError("incomplete inference output")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=ROOT / "training/data/futo-fresh")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "training/data/memory-wide")
    parser.add_argument("--frozen", type=Path, default=ROOT / "training/data/reranker/frozen-test.jsonl")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--challenger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        if args.frozen.exists():
            parser.error("frozen test file already exists; preserve it")
        freeze(args)
        return
    if not args.baseline or not args.challenger or not args.output:
        parser.error("comparison needs --baseline, --challenger, and --output")
    rows = [json.loads(line) for line in args.frozen.read_text().splitlines()]
    frozen_hash = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    manifest = json.loads(args.frozen.with_suffix(".manifest.json").read_text())
    if frozen_hash != manifest["sha256"]:
        raise ValueError("frozen evaluation data changed")
    runs = {name: predict(directory, args.frozen, len(rows)) for name, directory in [("baseline", args.baseline), ("challenger", args.challenger)]}
    report = {"vocabulary": 60000, "test_manifest": manifest, "groups": {}}
    rng = np.random.default_rng(112358)
    for group in sorted({r["group"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["group"] == group]
        metrics, correct = {}, {}
        for name, results in runs.items():
            hits = np.asarray([[p["word"] == rows[i]["word"] for p in results[i]["predictions"]] for i in indices])
            correct[name] = hits[:, 0].astype(float)
            times = [results[i]["milliseconds"] for i in indices]
            metrics[name] = {"top1": float(hits[:, 0].mean()), "top5": float(hits.any(axis=1).mean()),
                             "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
        difference = correct["challenger"] - correct["baseline"]
        clusters = defaultdict(list)
        for j, i in enumerate(indices):
            clusters[rows[i]["cluster"]].append(j)
        buckets = list(clusters.values())
        sums = np.asarray([difference[b].sum() for b in buckets])
        sizes = np.asarray([len(b) for b in buckets])
        draws = rng.integers(len(buckets), size=(4000, len(buckets)))
        estimates = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        report["groups"][group] = {"samples": len(indices), "clusters": len(buckets), **metrics,
            "top1_gain": float(difference.mean()), "cluster_bootstrap_95pct": np.quantile(estimates, [0.025, 0.975]).tolist()}
    report["models"] = {name: {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in directory.iterdir() if file.suffix in [".bin", ".json"]}
                        for name, directory in [("baseline", args.baseline), ("challenger", args.challenger)]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
