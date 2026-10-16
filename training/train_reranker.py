#!/usr/bin/env python3
"""Train a candidate ranker with listwise loss and validation-only selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, normalized, raw):
        return -raw[..., 1] / 0.1 + self.layers(normalized).squeeze(-1)


def load(path, device):
    saved = np.load(path, allow_pickle=False)
    return {key: torch.tensor(saved[key], device=device) for key in ["features", "candidates", "labels"]}, saved["groups"]


@torch.no_grad()
def evaluate(model, data, groups, mean, scale):
    logits = torch.cat([model(((batch - mean) / scale).clamp(-12, 12), batch) for batch in data["features"].split(512)])
    order = logits.topk(5, dim=1).indices
    ranked = data["candidates"].gather(1, order)
    result = {}
    for name in np.unique(groups):
        mask = torch.tensor(groups == name, device=logits.device)
        truth = data["labels"][mask, None]
        result[name] = {"top1": (ranked[mask, :1] == truth).any(dim=1).float().mean().item(),
                        "top5": (ranked[mask] == truth).any(dim=1).float().mean().item()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train, _ = load(args.train, device)
    validation, groups = load(args.validation, device)
    raw = train["features"]
    # Statistics use training candidates only, including difficult negatives.
    mean = raw.mean(dim=(0, 1))
    scale = raw.std(dim=(0, 1)).clamp_min(0.01)
    x = ((raw - mean) / scale).clamp(-12, 12)
    positive = train["candidates"] == train["labels"][:, None]
    eligible = positive.any(dim=1).nonzero().flatten()
    targets = positive.float().argmax(dim=1)
    model = Ranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00003)
    baseline = evaluate(model, validation, groups, mean, scale)
    best = sum(m["top1"] for m in baseline.values()) / len(baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    def export(metrics, epoch):
        layers = [{"weights": layer.weight.detach().cpu().tolist(), "bias": layer.bias.detach().cpu().tolist()} for layer in model.layers if isinstance(layer, nn.Linear)]
        model_json = {"schema_version": 1, "mean": mean.cpu().tolist(), "scale": scale.cpu().tolist(), "layers": layers,
                      "residual_weight": -0.1, "shortlist": 32}
        (args.output / "reranker.json").write_text(json.dumps(model_json, separators=(",", ":")) + "\n")
        (args.output / "metrics.json").write_text(json.dumps({"epoch": epoch, "baseline": baseline, "validation": metrics, "score": best}, indent=2) + "\n")
    export(baseline, 0)
    print(json.dumps({"baseline": baseline, "eligible_training_queries": len(eligible)}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = eligible[torch.randperm(len(eligible), device=device)]
        loss_sum = 0.0
        for indices in order.split(256):
            logits = model(x[indices], raw[indices])
            loss = F.cross_entropy(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        scheduler.step()
        model.eval()
        metrics = evaluate(model, validation, groups, mean, scale)
        score = sum(m["top1"] for m in metrics.values()) / len(metrics)
        result = {"epoch": epoch + 1, "loss": loss_sum / len(eligible), "validation": metrics, "score": score}
        print(json.dumps(result), flush=True)
        with (args.output / "history.jsonl").open("a") as log:
            log.write(json.dumps(result) + "\n")
        if score > best:
            best = score
            export(metrics, epoch + 1)
    report = {"seed": args.seed, "epochs": args.epochs, "test_data_used": False,
              "selection": "mean top1 across four validation sources; shortlist fixed at 32",
              "source_sha256": {"train": hashlib.sha256(args.train.read_bytes()).hexdigest(), "validation": hashlib.sha256(args.validation.read_bytes()).hexdigest()},
              "native_sources": {name: json.loads(path.with_suffix(".json").read_text()) for name, path in [("train", args.train), ("validation", args.validation)]}}
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
