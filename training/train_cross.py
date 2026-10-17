#!/usr/bin/env python3
"""Train cross-attention ranking on fresh writers and naturally retrieved rivals."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from character_model import character_targets
from cross_model import RouteCrossRanker
from motor_sampler import MotorSampler
from retrieval_model import build_encoder, load_recordings
from train_character import encode
from train_proxy import corpus, proxy_scores

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, validation, keys, lengths):
    hits = []
    for start in range(0, len(validation["labels"]), 8):
        part = {name: values[start:start + 8] for name, values in validation.items()}
        scores = model(part["sequence"], part["raw"], part["embedding"], part["candidates"], part["similarity"], keys, lengths)
        ranked = part["candidates"].gather(1, scores.argsort(dim=1, descending=True, stable=True)[:, :5])
        matches = ranked == part["labels"][:, None]
        hits.append(torch.stack([matches[:, 0], matches.any(1)], dim=1))
    hits = torch.cat(hits).float()
    metrics = {}
    for i, name in enumerate(["clean", "loose", "memory"]):
        mask = validation["difficulty"] == i
        metrics[name] = {"samples": int(mask.sum()), "top1": float(hits[mask, 0].mean()), "top5": float(hits[mask, 1].mean())}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--route-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=900347)
    args = parser.parse_args()
    if not 2 <= args.candidates <= 128:
        parser.error("candidate count must be between 2 and 128")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = "mps"
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["large", "intrinsic", "giant"]:
        raise ValueError("cross attention requires a residual backbone")
    encoder = build_encoder(architecture).to(device).eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    proxies = saved["proxies"].to(device)
    words = saved["words"]
    keys, lengths, _ = character_targets(words)
    keys, lengths = keys.to(device), lengths.to(device)
    sampler = MotorSampler(words, device)
    source = ROOT / "training/data/memory-proxy"
    train_rows, train_info = corpus(source / "train.bin")
    val_rows, val_info = corpus(source / "validation.bin")
    for name, info in [("train", train_info), ("validation", val_info)]:
        if info["sha256"] != saved["metadata"][name + "_corpus"]["sha256"]:
            raise ValueError("cross attention must preserve the encoder's declared data splits")
    train_paths = torch.tensor(np.array(train_rows["points"]), device=device)
    train_labels = torch.tensor(np.array(train_rows["label"], dtype=np.int64), device=device)
    selected = np.random.default_rng(args.seed).choice(len(val_rows), 3000, replace=False)
    val_paths = torch.tensor(np.array(val_rows["points"][selected]), device=device)
    validation = {"labels": torch.tensor(np.array(val_rows["label"][selected], dtype=np.int64), device=device),
        "difficulty": torch.tensor(np.array(val_rows["difficulty"][selected], dtype=np.int64), device=device)}
    with torch.no_grad():
        encoded = [encode(encoder, part) for part in val_paths.split(256)]
        for i, name in enumerate(["sequence", "raw", "embedding"]):
            validation[name] = torch.cat([part[i] for part in encoded])
        ranked = [proxy_scores(part, proxies).topk(128, dim=1) for part in validation["embedding"].split(256)]
        validation.update(similarity=torch.cat([r.values for r in ranked]), candidates=torch.cat([r.indices for r in ranked]))
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, {word: i for i, word in enumerate(words)})
    real, real_labels = torch.tensor(real, device=device), torch.tensor(real_labels, device=device)
    model = RouteCrossRanker(getattr(encoder, "input_channels", 8), route_geometry=args.route_geometry).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"encoder_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "encoder_path": str(args.checkpoint),
        "train": train_info, "validation": val_info, "test_data_used": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "real_training_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "selection": "Exact-word hard development top1; naturally retrieved top128; no target injection at evaluation"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.eval()
    print(json.dumps({"initial_validation": evaluate(model, validation, keys, lengths)}), flush=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        began, accumulated = time.monotonic(), 0.0
        for step in range(args.steps):
            nr, ns = args.batch // 5, args.batch // 5
            nf = args.batch - nr - ns
            ri = torch.randint(len(real), (nr,), device=device)
            si = torch.randint(len(train_paths), (ns,), device=device)
            fresh = sampler.labels(nf)
            paths = torch.cat([real[ri], train_paths[si], sampler.sample(fresh)])
            labels = torch.cat([real_labels[ri], train_labels[si], fresh])
            with torch.no_grad():
                sequence, raw, embedding = encode(encoder, paths)
                similarity = proxy_scores(embedding, proxies)
                pool = similarity.masked_fill(torch.arange(len(words), device=device)[None] == labels[:, None], -torch.inf).topk(128, dim=1).indices
                priority = torch.rand(pool.shape, device=device)
                priority[:, :16] += 1
                columns = priority.topk(args.candidates - 1, dim=1).indices
                candidates = torch.cat([labels[:, None], pool.gather(1, columns)], dim=1)
                prior = similarity.gather(1, candidates)
            scores = model(sequence, raw, embedding, candidates, prior, keys, lengths)
            loss = F.cross_entropy(scores, torch.zeros(args.batch, dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite cross-attention loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            accumulated += loss.item()
        scheduler.step()
        model.eval()
        metrics = evaluate(model, validation, keys, lengths)
        score = metrics["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        report = {"epoch": epoch + 1, "loss": accumulated / args.steps, "validation": metrics, "best_score": best,
            "improved": improved, "seconds": time.monotonic() - began}
        print(json.dumps(report), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save({"model": model.state_dict(), "metadata": metadata, "epoch": epoch,
                "raw_channels": getattr(encoder, "input_channels", 8)}, temporary)
            temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
