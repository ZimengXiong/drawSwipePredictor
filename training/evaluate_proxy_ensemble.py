#!/usr/bin/env python3
"""Compare fixed word-model ensembles on declared development writers only."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from retrieval_model import build_encoder, encoder_features
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= len(args.checkpoint) <= 4:
        parser.error("compare two to four models")
    torch.set_num_threads(4)
    rows, info = corpus(args.data)
    models, reports = [], []
    words = None
    for path in args.checkpoint:
        content = path.read_bytes()
        saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        if info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
            raise ValueError("ensemble selection requires the declared development writers")
        if words is not None and saved["words"] != words:
            raise ValueError("ensemble vocabularies differ")
        words = saved["words"]
        architecture = saved["metadata"]["arguments"].get("architecture", "small")
        encoder = build_encoder(architecture).to("mps").eval()
        encoder.load_state_dict(saved["model"])
        encoder.requires_grad_(False)
        models.append((encoder, saved["proxies"].to("mps")))
        reports.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "architecture": architecture, "epoch": saved["epoch"] + 1})
    count = len(models)
    weights = [tuple(float(i == j) for i in range(count)) for j in range(count)]
    weights.append(tuple(1 / count for _ in models))
    for a in range(count):
        for b in range(a + 1, count):
            for fraction in [0.25, 0.5, 0.75]:
                weights.append(tuple(fraction if i == a else 1 - fraction if i == b else 0 for i in range(count)))
    weights = list(dict.fromkeys(weights))
    paths = torch.tensor(np.array(rows["points"]), device="mps")
    labels = torch.tensor(np.array(rows["label"], dtype=np.int64), device="mps")
    difficulties = torch.tensor(np.array(rows["difficulty"], dtype=np.int64), device="mps")
    totals = torch.zeros(len(weights), 3, 4, device="mps")
    with torch.no_grad():
        for start in range(0, len(paths), 192):
            part = paths[start:start + 192]
            logits = [proxy_scores(encoder(encoder_features(encoder, part)), proxies) for encoder, proxies in models]
            for trial, mixing in enumerate(weights):
                combined = sum(weight * value for weight, value in zip(mixing, logits) if weight)
                ranked = combined.topk(128, dim=1).indices
                truth = labels[start:start + len(part), None]
                hits = torch.stack([(ranked[:, :k] == truth).any(1) for k in [1, 5, 32, 128]], dim=1).float()
                for difficulty in range(3):
                    mask = difficulties[start:start + len(part)] == difficulty
                    totals[trial, difficulty] += hits[mask].sum(0)
    trials = []
    for mixing, correct in zip(weights, totals.cpu()):
        metrics = {}
        for i, name in enumerate(["clean", "loose", "memory"]):
            size = int((difficulties == i).sum())
            metrics[name] = {"samples": size, **dict(zip(["top1", "top5", "top32", "top128"], (correct[i] / size).tolist()))}
        trials.append({"weights": mixing, "validation": metrics})
    report = {"checkpoints": reports, "development": info, "test_data_used": False,
        "selection": "Full-vocabulary exact-word memory development top1", "trials": trials,
        "selected": max(trials, key=lambda row: row["validation"]["memory"]["top1"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selected"]), flush=True)


if __name__ == "__main__":
    main()
