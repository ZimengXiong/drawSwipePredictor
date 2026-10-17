#!/usr/bin/env python3
"""Evaluate latent motor fitting on the declared development corpus only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from generative_fit import MotorFitter
from retrieval_model import build_encoder, encoder_features
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fixed-calibration", type=Path)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if min(args.samples, args.batch) < 1 or min(args.offset, args.steps) < 0:
        parser.error("use positive sample and batch counts; offset and steps may be zero")
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows, info = corpus(args.data)
    if info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("motor fitting experiments require the declared development corpus")
    selected = np.flatnonzero(rows["difficulty"] == 2)[args.offset:args.offset + args.samples]
    if len(selected) != args.samples:
        raise ValueError("requested development slice is unavailable")
    fixed = None
    if args.fixed_calibration:
        fixed = json.loads(args.fixed_calibration.read_text())
        if fixed["development_sha256"] != info["sha256"] or fixed["frozen_retriever_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError("fixed motor calibration sources differ")
        earlier = fixed["writer_selection"]
        if not (args.offset >= earlier["offset"] + earlier["samples"] or args.offset + args.samples <= earlier["offset"]):
            raise ValueError("fixed-calibration development slices overlap")
    paths = torch.tensor(np.array(rows["points"][selected]), device="mps")
    labels = torch.tensor(np.array(rows["label"][selected], dtype=np.int64), device="mps")
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    proxies = saved["proxies"].to("mps")
    fitter = MotorFitter(saved["words"])
    initializer, initializer_hash = None, None
    if args.initializer:
        from amortized_motor import AmortizedMotor
        from train_joint import load_checkpoint
        initial, initializer_hash = load_checkpoint(args.initializer)
        if initial["words"] != saved["words"] or initial["max_keys"] != fitter.keys.shape[1]:
            raise ValueError("motor initializer vocabulary or route width differs")
        initializer = AmortizedMotor(initial["max_keys"]).to("mps").eval()
        initializer.load_state_dict(initial["model"])
        initializer.requires_grad_(False)
    if fixed and fixed.get("initializer_sha256") != initializer_hash:
        raise ValueError("fixed motor calibration initializer differs")
    results = {key: [] for key in ["candidates", "similarity", "errors", "penalties"]}
    began = time.monotonic()
    for start in range(0, len(paths), args.batch):
        query = paths[start:start + args.batch]
        with torch.no_grad():
            similarity, candidates = proxy_scores(encoder(encoder_features(encoder, query)), proxies).topk(args.candidates, dim=1)
        errors, penalties = fitter.fit(query, candidates, steps=args.steps, initializer=initializer)
        for key, value in zip(results, [candidates, similarity, errors, penalties]):
            results[key].append(value.detach().cpu())
        print(json.dumps({"finished": start + len(query), "total": len(paths), "seconds": time.monotonic() - began}), flush=True)
    results = {key: torch.cat(value) for key, value in results.items()}
    labels = labels.cpu()
    candidates, similarity = results["candidates"], results["similarity"]
    trials = []
    for regularization in ([fixed["selected"]["regularization"]] if fixed else [0.0, 0.0002, 0.001, 0.005, 0.02]):
        cost = (results["errors"] + regularization * results["penalties"]).min(2).values
        for weight in ([fixed["selected"]["proxy_weight"]] if fixed else [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
            scores = -cost + weight * similarity
            ranked = candidates.gather(1, scores.argsort(dim=1, descending=True, stable=True))
            trials.append({"regularization": regularization, "proxy_weight": weight,
                "top1": float((ranked[:, 0] == labels).float().mean()),
                "top5": float((ranked[:, :5] == labels[:, None]).any(1).float().mean())})
    report = {"development_sha256": info["sha256"], "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "test_data_used": False, "calibration_frozen": fixed is not None,
        "calibration_sha256": hashlib.sha256(args.fixed_calibration.read_bytes()).hexdigest() if fixed else None,
        "initializer_sha256": initializer_hash,
        "optimization_regularization": 0.0002, "renderer_sha256": hashlib.sha256(Path(__file__).with_name("generative_fit.py").read_bytes()).hexdigest(),
        "samples": len(paths), "baseline_top1": float((candidates[:, 0] == labels).float().mean()),
        "shortlist_recall": float((candidates == labels[:, None]).any(1).float().mean()),
        "best": max(trials, key=lambda row: row["top1"]), "trials": trials,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({**results, "labels": labels, "metadata": report}, args.output.with_suffix(".pt"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
