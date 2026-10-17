#!/usr/bin/env python3
"""Build independent reference drawings and select their development scores."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from motor_sampler import MotorSampler
from retrieval_model import build_encoder, encoder_features
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def mixture_similarity(similarities, temperature):
    """Average equal-count reference kernels without favoring larger banks."""
    if temperature < 0:
        raise ValueError("reference temperature must be nonnegative")
    if temperature == 0:
        return similarities.max(-1).values
    return temperature * (torch.logsumexp(similarities / temperature, dim=-1) - math.log(similarities.shape[-1]))


def build(args):
    saved, source_hash = load_checkpoint(args.checkpoint)
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    words = saved["words"]
    sampler = MotorSampler(words, "mps", max_keys=None)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "vectors.tmp.npy"
    bank = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(words), args.references, 128))
    batch_words = max(1, args.batch // args.references)
    began = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(words), batch_words):
            end = min(start + batch_words, len(words))
            labels = torch.arange(start, end, device="mps").repeat_interleave(args.references)
            # Every dictionary word receives the same error-strength mixture.
            index = torch.arange(args.references, device="mps") / args.references
            strengths = torch.where(index < 0.125, 0.0, torch.where(index < 0.375, 0.55, 1.0)).repeat(end - start)
            paths = sampler.sample(labels, strength=strengths)
            vectors = encoder(encoder_features(encoder, paths))
            bank[start:end] = vectors.reshape(end - start, args.references, 128).cpu().numpy()
            if start == 0 or end == len(words) or start // 5000 != end // 5000:
                print(json.dumps({"words_finished": end, "total": len(words), "seconds": time.monotonic() - began}), flush=True)
    bank.flush()
    del bank
    destination = args.output / "vectors.npy"
    path.replace(destination)
    manifest = {"encoder_sha256": source_hash, "words": words, "references_per_word": args.references,
        "vectors_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "seed": args.seed, "batch": args.batch, "test_data_used": False,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["motor_sampler.py", "reference_density.py"]},
        "policy": "Equal reference count and fixed strength mixture for every full-dictionary word, including short and long words; no development drawings used to build the bank",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "build_seconds": time.monotonic() - began}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate(args):
    saved, source_hash = load_checkpoint(args.checkpoint)
    manifest = json.loads((args.bank / "manifest.json").read_text())
    if manifest["encoder_sha256"] != source_hash or manifest["words"] != saved["words"] or manifest["test_data_used"]:
        raise ValueError("reference bank belongs to a different encoder or dictionary")
    bank_path = args.bank / "vectors.npy"
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != manifest["vectors_sha256"]:
        raise ValueError("reference bank hash differs")
    vectors = np.load(bank_path, mmap_mode="r")
    if vectors.shape != (len(saved["words"]), manifest["references_per_word"], 128):
        raise ValueError("reference bank dimensions differ")
    bank = torch.tensor(np.array(vectors), device="mps")
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    proxies = saved["proxies"].to("mps")
    rows, info = corpus(args.data)
    if info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("reference selection requires the declared development corpus")
    temperatures = [0.0, 0.005, 0.01, 0.03, 0.1]
    cached = {key: [] for key in ["candidates", "similarity", "density"]}
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            paths = torch.tensor(np.array(rows["points"][start:start + 32]), device="mps")
            query = encoder(encoder_features(encoder, paths))
            similarity, candidates = proxy_scores(query, proxies).topk(128, dim=1)
            references = F.normalize(bank[candidates].float(), dim=-1)
            cosine = (query[:, None, None] * references).sum(-1)
            cached["density"].append(torch.stack([mixture_similarity(cosine, t) for t in temperatures], dim=-1).cpu())
            cached["candidates"].append(candidates.cpu())
            cached["similarity"].append(similarity.cpu())
    cached = {key: torch.cat(value) for key, value in cached.items()}
    labels = torch.tensor(np.array(rows["label"], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows["difficulty"], dtype=np.int64))
    trials = []
    for i, temperature in enumerate(temperatures):
        for fraction in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            if fraction == 0 and i:
                continue
            scores = (1 - fraction) * cached["similarity"] + fraction * cached["density"][:, :, i]
            ranked = cached["candidates"].gather(1, scores.argsort(dim=1, descending=True, stable=True)[:, :5])
            matches = ranked == labels[:, None]
            metrics = {}
            for level, name in enumerate(["clean", "loose", "memory"]):
                mask = difficulty == level
                metrics[name] = {"samples": int(mask.sum()), "top1": float(matches[mask, 0].float().mean()),
                                 "top5": float(matches[mask].any(1).float().mean())}
            trials.append({"temperature": temperature, "reference_fraction": fraction, "validation": metrics})
    report = {"encoder_sha256": source_hash, "bank_sha256": manifest["vectors_sha256"],
        "references_per_word": manifest["references_per_word"], "full_vocabulary": len(saved["words"]),
        "test_data_used": False, "development": info, "trials": trials,
        "selected": max(trials, key=lambda row: row["validation"]["memory"]["top1"]),
        "rules": "All development rows; natural top128 from full dictionary; exact-word scoring; no target insertion"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({**cached, "labels": labels, "difficulty": difficulty, "metadata": report}, args.output.with_suffix(".pt"))
    print(json.dumps(report["selected"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--checkpoint", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--references", type=int, default=32)
    builder.add_argument("--batch", type=int, default=512)
    builder.add_argument("--seed", type=int, default=900433)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--bank", type=Path, required=True)
    evaluator.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "build":
        if min(args.references, args.batch) < 1:
            parser.error("reference and batch counts must be positive")
        torch.manual_seed(args.seed)
        build(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
