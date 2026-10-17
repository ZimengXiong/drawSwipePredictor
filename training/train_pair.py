#!/usr/bin/env python3
"""Train local correspondence ranking with naturally retrieved validation lists."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from pair_model import PairRanker, pair_features
from retrieval_model import features, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def load(path, device):
    saved = np.load(path, allow_pickle=False)
    data = {key: torch.tensor(saved[key], device=device, dtype=torch.long if key in ["labels", "candidates", "difficulty"] else torch.float32)
            for key in ["paths", "labels", "candidates", "distances", "positive_distance", "difficulty"]}
    with torch.no_grad():
        data["query"] = torch.cat([pair_features(features(part)) for part in data.pop("paths").split(1024)])
    return data


@torch.no_grad()
def evaluate(model, data, templates):
    hits = []
    for start in range(0, len(data["query"]), 16):
        candidates = data["candidates"][start:start + 16]
        logits = model(data["query"][start:start + 16], templates[candidates], data["distances"][start:start + 16])
        ranked = candidates.gather(1, logits.topk(5, dim=1).indices)
        truth = data["labels"][start:start + 16, None]
        hits.append(torch.stack([(ranked[:, :1] == truth).any(1), (ranked == truth).any(1)], 1))
    hits = torch.cat(hits).float()
    result = {}
    for i, name in enumerate(["clean", "loose", "memory"]):
        selected = data["difficulty"] == i
        top1, top5 = hits[selected].mean(0).tolist()
        result[name] = {"samples": int(selected.sum()), "top1": top1, "top5": top5}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=673312)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if min(args.batch, args.steps, args.epochs) < 1 or not 2 <= args.candidates <= 128:
        parser.error("use positive counts and 2 to 128 candidates")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    reports = {name: json.loads(path.with_suffix(".json").read_text()) for name, path in [("train", args.train), ("validation", args.validation)]}
    for name, report in reports.items():
        if report["split"] != name or report["test_data_used"]:
            raise ValueError("pair fitting must preserve train and validation roles")
    if reports["train"]["retriever_sha256"] != reports["validation"]["retriever_sha256"]:
        raise ValueError("candidate pools require the same frozen retriever")
    train, validation = load(args.train, args.device), load(args.validation, args.device)
    _, _, paths = vocabulary(ROOT / "word_freq.txt", 60000)
    with torch.no_grad():
        templates = pair_features(features(torch.tensor(paths, device=args.device)))
    model = PairRanker().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00002)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"seed": args.seed, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "mining": reports, "test_data_used": False,
                "selection": "memory development top1 with unchanged naturally retrieved top128 lists",
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [("train", args.train), ("validation", args.validation)]}}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best = -1.0
    model.eval()
    print(json.dumps({"initial_validation": evaluate(model, validation, templates)}), flush=True)
    for epoch in range(args.epochs):
        began = time.monotonic()
        model.train()
        loss_sum = 0.0
        for step in range(args.steps):
            indices = torch.randint(len(train["query"]), (args.batch,), device=args.device)
            pool = train["candidates"][indices]
            choose = torch.rand(pool.shape, device=args.device)
            choose[:, :8] += 1.0
            choose.masked_fill_(pool == train["labels"][indices, None], -torch.inf)
            columns = choose.topk(args.candidates - 1, dim=1).indices
            candidate_ids = torch.cat([train["labels"][indices, None], pool.gather(1, columns)], dim=1)
            distances = torch.cat([train["positive_distance"][indices, None], train["distances"][indices].gather(1, columns)], dim=1)
            logits = model(train["query"][indices], templates[candidate_ids], distances)
            loss = F.cross_entropy(logits, torch.zeros(args.batch, dtype=torch.long, device=args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite pair loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        metrics = evaluate(model, validation, templates)
        score = metrics["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        result = {"epoch": epoch + 1, "loss": loss_sum / args.steps, "validation": metrics,
                  "seconds": time.monotonic() - began, "best_score": best, "improved": improved}
        print(json.dumps(result), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(result) + "\n")
        if improved:
            torch.save({"model": model.state_dict(), "metadata": metadata, "epoch": epoch, "best_score": best}, args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
