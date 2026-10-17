#!/usr/bin/env python3
"""Fit word-specific error covariance using independent synthetic references."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import build_encoder, encoder_features
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def factor_covariance(references, standard_deviation):
    """Factor within-word covariance without constructing a 128-square inverse."""
    references = references / standard_deviation
    means = references.mean(1)
    centered = (references - means[:, None]) / math.sqrt(references.shape[1] - 1)
    eigenvalues, directions = torch.linalg.eigh(centered @ centered.transpose(1, 2))
    factors = directions.transpose(1, 2) @ centered
    return means, factors, eigenvalues.clamp_min(0)


def gaussian_costs(query, candidates, means, factors, eigenvalues, standard_deviation, regularizers):
    """Evaluate regularized Gaussian costs with the Woodbury identity."""
    delta = query[:, None] / standard_deviation - means[candidates]
    squared = delta.square().sum(-1)
    projected = (factors[candidates] * delta[:, :, None]).sum(-1).square()
    values = eigenvalues[candidates]
    costs = []
    for scale in regularizers:
        mahalanobis = squared / scale - (projected / (scale * (scale + values))).sum(-1)
        log_determinant = torch.log1p(values / scale).sum(-1) + query.shape[-1] * math.log(scale)
        costs.append(0.5 * (mahalanobis + log_determinant) / query.shape[-1])
    return torch.stack(costs, dim=-1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    reference_manifest = json.loads((args.references / "manifest.json").read_text())
    if reference_manifest["test_data_used"] or reference_manifest["references_per_word"] < 2:
        raise ValueError("covariance fitting requires independent repeated references")
    path = args.references / "vectors.npy"
    if digest(path) != reference_manifest["vectors_sha256"]:
        raise ValueError("reference bank hash differs")
    vectors = np.load(path, mmap_mode="r")
    words, count, dimension = vectors.shape
    if words != len(reference_manifest["words"]) or count != reference_manifest["references_per_word"]:
        raise ValueError("reference bank dimensions differ")
    args.output.mkdir(parents=True, exist_ok=False)
    variance = torch.zeros(dimension, dtype=torch.float64)
    for start in range(0, words, 256):
        references = F.normalize(torch.tensor(np.array(vectors[start:start + 256]), dtype=torch.float32), dim=-1)
        variance += (references - references.mean(1, keepdim=True)).double().square().sum((0, 1))
    deviation = (variance / (words * (count - 1))).clamp_min(1e-6).sqrt().float()
    arrays = {"means": (words, dimension), "factors": (words, count, dimension), "eigenvalues": (words, count)}
    outputs = {key: np.lib.format.open_memmap(args.output / (key + ".npy"), mode="w+", dtype=np.float32, shape=shape)
               for key, shape in arrays.items()}
    began = time.monotonic()
    with torch.no_grad():
        for start in range(0, words, 256):
            references = F.normalize(torch.tensor(np.array(vectors[start:start + 256]), dtype=torch.float32), dim=-1)
            fitted = factor_covariance(references, deviation)
            for key, value in zip(arrays, fitted):
                outputs[key][start:start + len(references)] = value.numpy()
            if start % (256 * 40) == 0:
                print(json.dumps({"words_fitted": start + len(references), "total": words}), flush=True)
    for array in outputs.values():
        array.flush()
    del outputs
    np.save(args.output / "standard_deviation.npy", deviation.numpy())
    manifest = {"encoder_sha256": reference_manifest["encoder_sha256"], "words": reference_manifest["words"],
                "reference_manifest_sha256": digest(args.references / "manifest.json"),
                "reference_vectors_sha256": reference_manifest["vectors_sha256"], "test_data_used": False,
                "references_per_word": count, "fit_seconds": time.monotonic() - began,
                "source_sha256": digest(Path(__file__)),
                "artifacts": {key + ".npy": digest(args.output / (key + ".npy")) for key in [*arrays, "standard_deviation"]},
                "policy": "Equal reference count for every dictionary word. All covariance parameters and shared variances use references only; no development drawings, word ranks, frequencies, or benchmark targets."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate(args):
    saved, source_hash = load_checkpoint(args.checkpoint)
    manifest = json.loads((args.bank / "manifest.json").read_text())
    if manifest["test_data_used"] or manifest["words"] != saved["words"] or manifest["encoder_sha256"] != source_hash:
        raise ValueError("covariance bank belongs to a different encoder or vocabulary")
    arrays = {}
    for name, expected in manifest["artifacts"].items():
        if digest(args.bank / name) != expected:
            raise ValueError("covariance artifact hash differs")
        values = np.load(args.bank / name, mmap_mode="r")
        if not np.isfinite(values).all():
            raise ValueError("covariance parameters are nonfinite")
        arrays[Path(name).stem] = torch.tensor(np.array(values), device="mps")
    rows, info = corpus(args.data)
    if saved["metadata"]["test_data_used"] or info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("covariance selection requires the declared development corpus")
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval().requires_grad_(False)
    encoder.load_state_dict(saved["model"])
    proxies = saved["proxies"].to("mps")
    regularizers = [0.25, 1.0, 4.0]
    cached = {key: [] for key in ["candidates", "similarity", "cost"]}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            paths = torch.tensor(np.array(rows["points"][start:start + 16]), device="mps")
            query = encoder(encoder_features(encoder, paths))
            similarity, candidates = proxy_scores(query, proxies).topk(128, dim=1)
            costs = gaussian_costs(query, candidates, **arrays, regularizers=regularizers)
            if not torch.isfinite(costs).all():
                raise ValueError("covariance scoring produced nonfinite costs")
            for key, value in zip(cached, [candidates, similarity, costs]):
                cached[key].append(value.cpu())
            if start % 1024 == 0:
                print(json.dumps({"rows_scored": start + len(paths), "total": len(rows)}), flush=True)
    cached = {key: torch.cat(values) for key, values in cached.items()}
    labels = torch.tensor(np.array(rows["label"], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows["difficulty"], dtype=np.int64))
    trials, predictions = [], []
    for i, regularizer in enumerate(regularizers):
        for fraction in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
            if i and fraction == 0:
                continue
            scores = (1 - fraction) * cached["similarity"] - fraction * cached["cost"][:, :, i]
            ranked = cached["candidates"].gather(1, scores.argsort(dim=1, descending=True, stable=True)[:, :5])
            hits = ranked == labels[:, None]
            metrics = {}
            for level, name in enumerate(["clean", "loose", "memory"]):
                mask = difficulty == level
                metrics[name] = {"samples": int(mask.sum()), "top1": float(hits[mask, 0].float().mean()),
                                 "top5": float(hits[mask].any(1).float().mean())}
            trials.append({"regularizer": regularizer, "covariance_fraction": fraction, "validation": metrics})
            predictions.append(hits[:, 0])
    best = max(range(len(trials)), key=lambda i: trials[i]["validation"]["memory"]["top1"])
    mask = difficulty == 2
    gained = int((predictions[best] & ~predictions[0] & mask).sum())
    lost = int((~predictions[best] & predictions[0] & mask).sum())
    report = {"encoder_sha256": source_hash, "bank_manifest_sha256": digest(args.bank / "manifest.json"),
              "test_data_used": False, "development": info, "full_vocabulary": len(saved["words"]),
              "selected": trials[best], "trials": trials, "paired_memory": {"gained": gained, "lost": lost, "net": gained - lost},
              "rules": "All development rows; natural top128 from full vocabulary; exact full words. Fitted covariance uses independent references only. No target insertion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["selected"], "paired_memory": report["paired_memory"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--references", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--bank", type=Path, required=True)
    evaluator.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    build(args) if args.mode == "build" else evaluate(args)


if __name__ == "__main__":
    main()
