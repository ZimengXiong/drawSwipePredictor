#!/usr/bin/env python3
"""Test real-training exposure correction on full development writers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from retrieval_model import build_encoder, encoder_features, load_recordings
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def exposure_bias(counts, balanced, real_fraction=0.15):
    """Compute a class offset from training exposure and a uniform floor."""
    if not 0 <= real_fraction < 1 or counts.ndim != 1 or not torch.isfinite(counts).all() or (counts < 0).any() or not counts.sum():
        raise ValueError("invalid real-training exposure counts or mixture")
    probability = (counts > 0).float() if balanced else counts.float()
    probability = probability / probability.sum()
    # Use the same procedural floor for every dictionary word. No evaluation
    # labels, word popularity, word lengths, or vocabulary ranks enter this bias.
    ratio = real_fraction * len(counts) * probability / (1 - real_fraction)
    return ratio.log1p() / 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--real-train", type=Path, default=Path("data/futo-large/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved, source_hash = load_checkpoint(args.checkpoint)
    rows, info = corpus(args.data)
    if saved["metadata"]["validation_corpus"]["sha256"] != info["sha256"]:
        raise ValueError("exposure selection requires the declared development corpus")
    real_hash = hashlib.sha256(args.real_train.read_bytes()).hexdigest()
    if saved["metadata"]["real_training_sha256"] != real_hash:
        raise ValueError("exposure counts require the checkpoint's real training data")
    words = saved["words"]
    _, labels = load_recordings(args.real_train, {word: i for i, word in enumerate(words)})
    counts = torch.bincount(torch.tensor(labels), minlength=len(words))
    balanced = saved["metadata"]["arguments"].get("balance_real", False)
    bias = exposure_bias(counts, balanced).to("mps")
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    proxies = saved["proxies"].to("mps")
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    totals = torch.zeros(len(strengths), 3, 2, device="mps")
    cached_predictions = [[] for _ in strengths]
    with torch.no_grad():
        for start in range(0, len(rows), 192):
            paths = torch.tensor(np.array(rows["points"][start:start + 192]), device="mps")
            truth = torch.tensor(np.array(rows["label"][start:start + 192], dtype=np.int64), device="mps")
            difficulty = torch.tensor(np.array(rows["difficulty"][start:start + 192], dtype=np.int64), device="mps")
            similarity = proxy_scores(encoder(encoder_features(encoder, paths)), proxies)
            for i, strength in enumerate(strengths):
                ranked = (similarity - strength * bias).topk(5, dim=1).indices
                hits = ranked == truth[:, None]
                cached_predictions[i].append(ranked[:, 0].cpu())
                for level in range(3):
                    mask = difficulty == level
                    totals[i, level, 0] += hits[mask, 0].sum()
                    totals[i, level, 1] += hits[mask].any(1).sum()
    trials = []
    for strength, correct in zip(strengths, totals.cpu()):
        metrics = {}
        for level, name in enumerate(["clean", "loose", "memory"]):
            count = int((rows["difficulty"] == level).sum())
            metrics[name] = {"samples": count, "top1": float(correct[level, 0] / count), "top5": float(correct[level, 1] / count)}
        trials.append({"strength": strength, "validation": metrics})
    selected = max(trials, key=lambda row: row["validation"]["memory"]["top1"])
    report = {"checkpoint_sha256": source_hash, "development": info, "test_data_used": False,
        "real_training_sha256": real_hash, "balanced_real_sampling": balanced, "full_vocabulary": len(words),
        "observed_training_word_types": int((counts > 0).sum()), "real_fraction": 0.15,
        "bias_sha256": hashlib.sha256(bias.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
        "rules": "All development rows and all dictionary scores; class offsets use training-recording counts only; no target insertion or candidate filtering",
        "limits": "Uniform-floor exposure hypothesis, not an exact correction for the whole training distribution; development calibration only",
        "selected": selected, "trials": trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"predictions": torch.stack([torch.cat(value) for value in cached_predictions]),
                "strengths": strengths, "bias": bias.cpu(), "metadata": report}, args.output.with_suffix(".pt"))
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
