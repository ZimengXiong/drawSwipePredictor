#!/usr/bin/env python3
"""Learn full-vocabulary word proxies from fresh synthetic writers and real traces."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retrieval_model import GestureEncoder, augment, features, load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]
RECORD = np.dtype([("label", "<u4"), ("difficulty", "<u4"), ("writer", "<u8"), ("points", "<f4", (64, 2))])


def corpus(path):
    with path.open("rb") as source:
        magic, count, size, seed = struct.unpack("<8sQIQ", source.read(28))
    if magic != b"SWPCOR01" or path.stat().st_size != 28 + count * RECORD.itemsize:
        raise ValueError("invalid synthetic corpus format")
    rows = np.memmap(path, dtype=RECORD, mode="r", offset=28, shape=(count,))
    if rows["label"].max() >= size or rows["difficulty"].max() > 2 or not np.isfinite(rows["points"]).all():
        raise ValueError("invalid synthetic corpus values")
    if not np.array_equal(rows["writer"], seed * 1_000_003 + np.arange(count, dtype=np.uint64)):
        raise ValueError("unexpected writer identities")
    return rows, {"samples": count, "vocabulary": size, "seed": seed,
                  "writer_start": seed * 1_000_003, "writer_end": seed * 1_000_003 + count,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@torch.no_grad()
def evaluate(model, proxies, datasets):
    bank = F.normalize(proxies, dim=1)
    result = {}
    for name, (paths, labels) in datasets.items():
        hits = []
        for start in range(0, len(paths), 256):
            ranked = (model(features(paths[start:start + 256])) @ bank.T).topk(5, dim=1).indices
            truth = labels[start:start + 256, None]
            hits.append(torch.stack([(ranked[:, :1] == truth).any(1), (ranked == truth).any(1)], 1))
        top1, top5 = torch.cat(hits).float().mean(0).tolist()
        result[name] = {"samples": len(paths), "top1": top1, "top5": top5}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "training/data/memory-proxy/train.bin")
    parser.add_argument("--validation", type=Path, default=ROOT / "training/data/memory-proxy/validation.bin")
    parser.add_argument("--init-model", type=Path, default=ROOT / "training/checkpoints/retrieval-v3/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--seed", type=int, default=812831)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.75)
    train, train_info = corpus(args.train)
    validation, validation_info = corpus(args.validation)
    if not (train_info["writer_end"] <= validation_info["writer_start"] or validation_info["writer_end"] <= train_info["writer_start"]):
        raise ValueError("training and validation writers overlap")
    for record in [train_info, validation_info]:
        if record["seed"] in [101, 102, 9001, 12345, 246810]:
            raise ValueError("reserved benchmark writer block")
    words, counts, canonical = vocabulary(ROOT / "word_freq.txt", train_info["vocabulary"])
    if len(words) != validation_info["vocabulary"]:
        raise ValueError("corpus vocabularies differ")
    ids = {word: i for i, word in enumerate(words)}
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, ids)
    real = torch.tensor(real, device=device)
    real_labels = torch.tensor(real_labels, device=device)
    train_paths = torch.tensor(np.array(train["points"]), device=device)
    train_labels = torch.tensor(np.array(train["label"], dtype=np.int64), device=device)
    canonical = torch.tensor(canonical, device=device)
    datasets = {}
    for difficulty, name in enumerate(["clean", "loose", "memory"]):
        mask = validation["difficulty"] == difficulty
        datasets[name] = (torch.tensor(np.array(validation["points"][mask]), device=device), torch.tensor(np.array(validation["label"][mask], dtype=np.int64), device=device))
    model = GestureEncoder().to(device)
    initial = torch.load(args.init_model, map_location=device, weights_only=False)
    model.load_state_dict(initial["model"])
    model.eval()
    with torch.no_grad():
        initial_vectors = torch.cat([model(features(part)) for part in canonical.split(512)])
    proxies = nn.Parameter(initial_vectors)
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": args.learning_rate},
                                  {"params": [proxies], "lr": args.learning_rate * 4}], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    start, best = 0, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        proxies.data.copy_(saved["proxies"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"] + 1, saved["best_score"]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_corpus": train_info, "validation_corpus": validation_info,
        "real_training_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "test_data_used": False, "selection": "uniform full-vocabulary memory validation top1",
        "input_features": "geometry only; labels and writer identities are never model inputs",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"initial_validation": evaluate(model, proxies, datasets)}), flush=True)
    for epoch in range(start, args.epochs):
        model.train()
        began, loss_sum = time.monotonic(), 0.0
        for step in range(args.steps):
            nr, nc = args.batch * 15 // 100, args.batch // 10
            ns = args.batch - nr - nc
            ri = torch.randint(len(real), (nr,), device=device)
            si = torch.randint(len(train_paths), (ns,), device=device)
            ci = torch.randint(len(words), (nc,), device=device)
            # Procedural drawings already contain strong memory errors. Avoid
            # destroying their remaining evidence with a second heavy distortion.
            paths = torch.cat([augment(real[ri]), train_paths[si], augment(canonical[ci])])
            labels = torch.cat([real_labels[ri], train_labels[si], ci])
            query = model(features(paths))
            logits = 32.0 * query @ F.normalize(proxies, dim=1).T
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite proxy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), proxies], 5)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        metrics = evaluate(model, proxies, datasets)
        score = metrics["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        result = {"epoch": epoch + 1, "loss": loss_sum / args.steps, "validation": metrics,
                  "best_score": best, "seconds": time.monotonic() - began, "improved": improved}
        print(json.dumps(result), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(result) + "\n")
        saved = {"model": model.state_dict(), "proxies": F.normalize(proxies.detach(), dim=1),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best_score": best, "words": words, "counts": counts, "metadata": metadata}
        torch.save(saved, args.output / "latest.pt")
        if improved:
            torch.save(saved, args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
