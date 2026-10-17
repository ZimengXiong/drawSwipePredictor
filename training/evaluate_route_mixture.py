#!/usr/bin/env python3
"""Check omission-prior scoring against cached development fits and neural evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from character_model import character_targets
from generative_fit import route_mixture_cost
from train_joint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["fits", "combination", "fit_retriever", "motor_calibration", "output"]:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fits = torch.load(args.fits, map_location="cpu", weights_only=False)
    combined = torch.load(args.combination, map_location="cpu", weights_only=False)
    retriever, retriever_hash = load_checkpoint(args.fit_retriever)
    _, lengths, _ = character_targets(retriever["words"])
    source = combined["metadata"]
    calibration = json.loads(args.motor_calibration.read_text())
    if source["test_data_used"] or fits["metadata"]["test_data_used"] or calibration["test_data_used"]:
        raise ValueError("route mixture requires development-only sources")
    if retriever_hash != fits["metadata"]["checkpoint_sha256"] or retriever_hash != source["fit_retriever_sha256"]:
        raise ValueError("route-mixture word banks differ")
    if not torch.equal(fits["labels"], combined["labels"]):
        raise ValueError("route-mixture rows differ")
    if calibration["development_sha256"] != source["development_sha256"]:
        raise ValueError("route-mixture development corpora differ")
    fixed = json.loads(args.calibration.read_text()) if args.calibration else None
    if fixed:
        for key in ["primary_sha256", "auxiliary_sha256", "fit_retriever_sha256", "development_sha256"]:
            if source[key] != fixed[key]:
                raise ValueError("fixed route-mixture sources differ")
        previous, current = fixed["writer_selection"], source["writer_selection"]
        if not (current["offset"] >= previous["offset"] + previous["samples"] or current["offset"] + current["samples"] <= previous["offset"]):
            raise ValueError("route-mixture development slices overlap")
    labels, candidates, valid, scores = [combined[key] for key in ["labels", "candidates", "valid", "scores"]]
    prior = 0.35
    regularization = calibration["selected"]["regularization"]
    blend = 32.0
    before = candidates[:, :128].gather(1, scores[:, :128].argmax(1)[:, None])[:, 0] == labels
    trials = []
    for temperature in ([fixed["selected"]["temperature"]] if fixed else [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01]):
        costs = route_mixture_cost(fits["errors"], fits["penalties"], lengths[fits["candidates"]], regularization, temperature, prior)
        if not torch.isfinite(costs).all():
            raise ValueError("nonfinite route scores; do not omit rows")
        middle = costs.median(1).values
        relative = (costs - middle[:, None]) / (middle - costs.min(1).values).clamp_min(1e-6)[:, None]
        evidence = torch.zeros_like(scores)
        for i in range(len(labels)):
            positions = {word: j for j, word in enumerate(candidates[i, valid[i]].tolist())}
            for j, word in enumerate(fits["candidates"][i].tolist()):
                evidence[i, positions[word]] = relative[i, j]
        order = (scores - blend * evidence).argsort(dim=1, descending=True, stable=True)[:, :5]
        hits = candidates.gather(1, order) == labels[:, None]
        trials.append({"temperature": temperature, "top1": float(hits[:, 0].float().mean()),
            "top5": float(hits.any(1).float().mean()), "gained": int((hits[:, 0] & ~before).sum()), "lost": int((~hits[:, 0] & before).sum())})
    report = {key: source[key] for key in ["primary_sha256", "auxiliary_sha256", "fit_retriever_sha256", "development_sha256", "writer_selection"]}
    report.update(test_data_used=False, calibration_frozen=fixed is not None, omission_prior=prior, regularization=regularization,
        motor_weight=blend, baseline_complete_top1=float(before.float().mean()), trials=trials,
        selected=max(trials, key=lambda row: row["top1"]),
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [("fits", args.fits), ("combination", args.combination), ("motor_calibration", args.motor_calibration)]},
        limits="Reused development data for component selection; not an untouched benchmark or confirmation result.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selected"]), flush=True)


if __name__ == "__main__":
    main()
