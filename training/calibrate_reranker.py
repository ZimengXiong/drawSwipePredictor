#!/usr/bin/env python3
"""Protect confident retrieval matches using a validation-selected fallback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=ROOT / "training/data/futo-large")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "training/data/memory-wide")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = np.load(args.validation, allow_pickle=False)
    canonical = np.load(args.canonical, allow_pickle=False)
    combined = {key: np.concatenate([saved[key], canonical[key]]) for key in ["features", "candidates", "labels", "groups"]}
    raw = torch.tensor(combined["features"])
    config = json.loads(args.model.read_text())
    x = ((raw - torch.tensor(config["mean"])) / torch.tensor(config["scale"])).clamp(-12, 12)
    for i, layer in enumerate(config["layers"]):
        x = F.linear(x, torch.tensor(layer["weights"]), torch.tensor(layer["bias"]))
        if i + 1 < len(config["layers"]):
            x = x.relu()
    cost = raw[:, :, 1] + config["residual_weight"] * x[:, :, 0]
    candidates, labels, groups = combined["candidates"], combined["labels"], combined["groups"]
    metadata = json.loads(args.validation.with_suffix(".json").read_text())
    count = int((groups == "real").sum())
    words, _, _ = vocabulary(ROOT / "word_freq.txt", 60000)
    ids = {word: i for i, word in enumerate(words)}
    lengths = {}
    for name, directory in [("real", args.real_dir), ("synthetic", args.synthetic_dir)]:
        path = directory / "validation.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["source_sha256"][name]:
            raise ValueError("validation recordings differ from candidate provenance")
        paths, _ = load_recordings(path, ids)
        lengths[name] = len(paths)
    chooser = np.random.default_rng(metadata["seed"])
    chooser.choice(lengths["real"], count, replace=False)
    indices = chooser.choice(lengths["synthetic"], count, replace=False)
    with (args.synthetic_dir / "validation.jsonl").open() as source:
        records = [json.loads(line) for line in source]
    if len(records) != lengths["synthetic"]:
        raise ValueError("difficulty mapping requires valid in-vocabulary synthetic rows")
    styles = np.asarray([r["difficulty"].lower() for r in records])[indices]
    expected_labels = np.asarray([ids[records[i]["word"]] for i in indices])
    if not np.array_equal(expected_labels, labels[groups == "synthetic"]):
        raise ValueError("validation sampler no longer matches extracted candidate lists")
    synthetic_rows = np.flatnonzero(groups == "synthetic")
    masks = {name: groups == name for name in np.unique(groups)}
    for name in ["clean", "loose", "memory"]:
        mask = np.zeros(len(groups), dtype=bool)
        mask[synthetic_rows[styles == name]] = True
        masks["synthetic_" + name] = mask
    base_order = raw[:, :, 1].argsort(dim=1)[:, :5].numpy()
    reranked = cost.argsort(dim=1)[:, :5].numpy()
    def metrics(order):
        hits = np.take_along_axis(candidates, order, axis=1) == labels[:, None]
        return {name: {"top1": float(hits[mask, 0].mean()), "top5": float(hits[mask].any(axis=1).mean())} for name, mask in masks.items()}
    baseline = metrics(base_order)
    trials = []
    best = None
    for threshold in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]:
        trusted = raw[:, 0, 0].numpy() < threshold
        order = np.where(trusted[:, None], base_order, reranked)
        result = metrics(order)
        score = sum(result[name]["top1"] for name in np.unique(groups)) / len(np.unique(groups))
        # Preserve clean recognition while selecting gains on the other sources.
        eligible = all(result[name]["top1"] >= baseline[name]["top1"] - 0.002 for name in ["synthetic_clean", "canonical"])
        trial = {"trust_distance": threshold, "score": score, "eligible": eligible, "validation": result}
        trials.append(trial)
        if eligible and (best is None or score > best["score"]):
            best = trial
    if best is None:
        raise ValueError("no fallback preserves clean validation accuracy")
    config["trust_distance"] = best["trust_distance"]
    args.output.write_text(json.dumps(config, separators=(",", ":")) + "\n")
    report = {"baseline": baseline, "selected": best, "trials": trials, "test_data_used": False,
              "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
              "canonical_sha256": hashlib.sha256(args.canonical.read_bytes()).hexdigest()}
    args.output.with_suffix(".calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "selected": best}, indent=2))


if __name__ == "__main__":
    main()
