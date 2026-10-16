#!/usr/bin/env python3
"""Select empirical word prototypes and a frequency prior on validation data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import GestureEncoder, augment, features, load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def encode(model, paths):
    return torch.cat([model(features(part)) for part in paths.split(512)])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=ROOT / "training/data/futo-large")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "training/data/memory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(161803)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = GestureEncoder().to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    words, counts, canonical_array = vocabulary(ROOT / "word_freq.txt", len(checkpoint["words"]))
    ids = {word: i for i, word in enumerate(words)}
    canonical_paths = torch.tensor(canonical_array, device=args.device)
    canonical = encode(model, canonical_paths)
    train_paths, train_labels = load_recordings(args.real_dir / "train.jsonl", ids)
    train_paths_gpu = torch.tensor(train_paths, device=args.device)
    train_vectors = encode(model, train_paths_gpu)
    train_labels = torch.tensor(train_labels, device=args.device)
    totals = torch.zeros_like(canonical).index_add_(0, train_labels, train_vectors)
    seen = torch.bincount(train_labels, minlength=len(words)).float().unsqueeze(1)
    means = F.normalize(totals, dim=1)
    confidence = seen / (seen + 3.0)
    popularity = torch.tensor(np.log1p(counts) / np.log1p(counts.max()), dtype=torch.float32, device=args.device)
    datasets = {}
    chooser = np.random.default_rng(161803)
    for name, path in [("real", args.real_dir / "validation.jsonl"), ("synthetic", args.synthetic_dir / "validation.jsonl")]:
        paths, labels = load_recordings(path, ids)
        select = chooser.choice(len(paths), min(args.samples, len(paths)), replace=False)
        paths = torch.tensor(paths[select], device=args.device)
        labels = torch.tensor(labels[select], device=args.device)
        datasets[name] = (paths, labels)
        if name == "real":
            datasets["distorted_real"] = (augment(paths, force_memory=True), labels)
    canonical_ids = torch.tensor(chooser.choice(len(words), min(args.samples, len(words)), replace=False), device=args.device)
    datasets["canonical"] = (torch.tensor(canonical_array, device=args.device)[canonical_ids], canonical_ids)
    best = None
    trials = []
    for tta in [0.0, 10.0]:
        embeddings = {}
        for name, (paths, labels) in datasets.items():
            vectors = encode(model, paths)
            if tta:
                angle = np.deg2rad(tta)
                for sign in [-1, 1]:
                    rotation = torch.tensor([[np.cos(angle), -sign * np.sin(angle)], [sign * np.sin(angle), np.cos(angle)]], dtype=torch.float32, device=args.device)
                    vectors += encode(model, paths @ rotation)
                vectors = F.normalize(vectors, dim=1)
            embeddings[name] = vectors
        if tta:
            # Use the same orientation ensemble for queries and stored routes.
            # Otherwise even an ideal route can differ from its own prototype.
            for source, name in [(canonical_paths, "canonical"), (train_paths_gpu, "train")]:
                combined = encode(model, source)
                for sign in [-1, 1]:
                    rotation = torch.tensor([[np.cos(angle), -sign * np.sin(angle)], [sign * np.sin(angle), np.cos(angle)]], dtype=torch.float32, device=args.device)
                    combined += encode(model, source @ rotation)
                combined = F.normalize(combined, dim=1)
                if name == "canonical":
                    canonical = combined
                else:
                    totals = torch.zeros_like(canonical).index_add_(0, train_labels, combined)
                    means = F.normalize(totals, dim=1)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weight = alpha * confidence
            prototypes = F.normalize(canonical * (1 - weight) + means * weight, dim=1)
            metrics = {(form, prior): {} for form, grid in [("linear", [0.0, 0.025, 0.05, 0.1, 0.2, 0.35]), ("adaptive", [0.1, 0.25, 0.5, 1.0, 2.0])] for prior in grid}
            for name, vectors in embeddings.items():
                labels = datasets[name][1]
                scores = vectors @ prototypes.T
                for form, prior in metrics:
                    ranked = scores + prior * popularity if form == "linear" else scores - prior * (1 - popularity) * (1 - scores).clamp_min(0).sqrt()
                    top = ranked.topk(5, dim=1).indices
                    metrics[(form, prior)][name] = {
                        "top1": (top[:, 0] == labels).float().mean().item(),
                        "top5": (top == labels[:, None]).any(dim=1).float().mean().item(),
                    }
            for (form, prior), accuracy in metrics.items():
                score = sum(m["top1"] + 0.05 * m["top5"] for m in accuracy.values()) / len(accuracy)
                trial = {"alpha": alpha, "prior": prior, "prior_form": form, "tta_degrees": tta, "score": score, "validation": accuracy}
                trials.append(trial)
                if best is None or score > best["score"]:
                    best = trial
            print(json.dumps({"tta": tta, "alpha": alpha, "best": best}), flush=True)
    weight = best["alpha"] * confidence
    prototypes = F.normalize(canonical * (1 - weight) + means * weight, dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), vectors=prototypes.cpu().numpy(), words=np.asarray(words))
    report = {
        "schema_version": 1, "selected": best, "trials": trials,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "real_training_samples": len(train_paths), "empirical_word_prototypes": int((seen > 0).sum().item()),
        "samples_per_validation_group": args.samples, "test_data_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
