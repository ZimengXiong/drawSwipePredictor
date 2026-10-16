#!/usr/bin/env python3
"""Train a contrastive gesture recognizer on real and synthetic motor traces."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import GestureEncoder, augment, features, load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def bank(model, canonical, batch=1024):
    return torch.cat([model(features(part)) for part in canonical.split(batch)])


@torch.no_grad()
def neighbors(embeddings, count=8):
    """Mine confusable word routes while bounding the similarity matrix size."""
    result = []
    for start in range(0, len(embeddings), 512):
        scores = embeddings[start:start + 512] @ embeddings.T
        scores[torch.arange(len(scores), device=scores.device), torch.arange(start, start + len(scores), device=scores.device)] = -2
        result.append(scores.topk(count, dim=1).indices)
    return torch.cat(result)


@torch.no_grad()
def evaluate(model, embeddings, datasets):
    result = {}
    for name, (paths, labels) in datasets.items():
        ranks = []
        for start in range(0, len(paths), 256):
            scores = model(features(paths[start:start + 256])) @ embeddings.T
            top = scores.topk(5, dim=1).indices
            truth = labels[start:start + 256, None]
            ranks.append(torch.stack([(top[:, :1] == truth).any(dim=1), (top == truth).any(dim=1)], dim=1))
        metrics = torch.cat(ranks).float().mean(dim=0).cpu().tolist()
        result[name] = {"top1": metrics[0], "top5": metrics[1], "samples": len(paths)}
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=ROOT / "training/data/futo-large")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "training/data/memory")
    parser.add_argument("--output", type=Path, default=ROOT / "training/checkpoints/retrieval")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--validation-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch, args.vocab, args.validation_samples) <= 0 or args.batch < 8:
        parser.error("counts must be positive and batch size must be at least 8")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.65)
    words, counts, canonical_array = vocabulary(ROOT / "word_freq.txt", args.vocab)
    word_ids = {word: i for i, word in enumerate(words)}
    canonical = torch.tensor(canonical_array, device=device)
    files = {
        "real_train": args.real_dir / "train.jsonl",
        "real_validation": args.real_dir / "validation.jsonl",
        "synthetic_train": args.synthetic_dir / "train.jsonl",
        "synthetic_validation": args.synthetic_dir / "validation.jsonl",
    }
    print("Loading recordings and resampling paths", flush=True)
    loaded = {}
    for name, path in files.items():
        paths, labels = load_recordings(path, word_ids)
        if "validation" in name:
            paths, labels = paths[:args.validation_samples], labels[:args.validation_samples]
        loaded[name] = (torch.tensor(paths, device=device), torch.tensor(labels, device=device))
    real, real_labels = loaded["real_train"]
    synthetic, synthetic_labels = loaded["synthetic_train"]
    val_real, val_labels = loaded["real_validation"]
    datasets = {
        "real": (val_real, val_labels),
        "distorted_real": (augment(val_real, force_memory=True), val_labels),
        "synthetic": loaded["synthetic_validation"],
    }
    model = GestureEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00002)
    args.output.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_score"]
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "device": str(device),
        "training_recordings": {"real": len(real), "synthetic": len(synthetic)},
        "source_sha256": {name: digest(path) for name, path in files.items()},
        "selection": "mean validation top1 over recorded real, distorted real, and synthetic gestures",
        "test_data_used": False,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hard = None
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.eval()
        if hard is None or epoch % 5 == 0:
            hard = neighbors(bank(model, canonical))
        model.train()
        total_loss = 0.0
        for step in range(args.steps):
            n_real, n_synthetic = args.batch // 4, args.batch // 4
            n_online = args.batch - n_real - n_synthetic
            real_indices = torch.randint(len(real), (n_real,), device=device)
            syn_indices = torch.randint(len(synthetic), (n_synthetic,), device=device)
            online_labels = torch.randint(len(words), (n_online,), device=device)
            labels = torch.cat([real_labels[real_indices], synthetic_labels[syn_indices], online_labels])
            query_paths = augment(torch.cat([real[real_indices], synthetic[syn_indices], canonical[online_labels]]))
            hard_ids = hard[labels, torch.randint(hard.shape[1], (len(labels),), device=device)]
            candidate_ids = torch.unique(torch.cat([labels, hard_ids, torch.randint(len(words), (256,), device=device)]))
            all_features = features(torch.cat([query_paths, canonical[candidate_ids]]))
            vectors = model(all_features)
            query, targets = vectors[:len(labels)], vectors[len(labels):]
            logits = query @ targets.T / 0.07
            positive = labels[:, None] == candidate_ids[None, :]
            loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"nonfinite loss at epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        metrics = evaluate(model, bank(model, canonical), datasets)
        score = sum(m["top1"] for m in metrics.values()) / len(metrics)
        improved = score > best
        best = max(best, score)
        result = {
            "epoch": epoch + 1, "loss": total_loss / args.steps, "validation": metrics,
            "score": score, "best_score": best, "seconds": time.monotonic() - started,
            "learning_rate": optimizer.param_groups[0]["lr"], "improved": improved,
        }
        print(json.dumps(result), flush=True)
        with (args.output / "history.jsonl").open("a") as log:
            log.write(json.dumps(result) + "\n")
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_score": best, "words": words, "counts": counts, "metadata": metadata,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if improved:
            torch.save(checkpoint, args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
