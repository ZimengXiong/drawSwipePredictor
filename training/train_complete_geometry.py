#!/usr/bin/env python3
"""Learn a class-independent geometry correction around complete predictions."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from prepare_complete_geometry import ROOT, FEATURES, SCHEMA, digest, model_inventory, validate_roles
from train_reranker import Ranker

DERIVED_SCHEMA = "complete_geometry_margin_v1"
BLENDS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def derived_features(raw):
    """Replace the reserved zero with a query-relative geometry margin."""
    if raw.ndim != 3 or raw.shape[-1] != 20 or not torch.isfinite(raw).all() or torch.any(raw[..., 2] != 0):
        raise ValueError("expected finite geometry-v1 features with an unused popularity slot")
    result = raw.clone()
    result[..., 2] = raw[..., 4] - raw[..., 4].amin(dim=1, keepdim=True)
    return result


def metrics(scores, candidates, labels, difficulty):
    """Count every query, including targets missing from the natural shortlist."""
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :5]
    hits = candidates.gather(1, order) == labels[:, None]
    result = {}
    for level, name in enumerate(["clean", "loose", "memory"]):
        mask = difficulty == level
        size = int(mask.sum())
        first, five = int(hits[mask, 0].sum()), int(hits[mask].any(1).sum())
        result[name] = {"samples": size, "correct_top1": first, "correct_top5": five,
                        "top1": first / max(size, 1), "top5": five / max(size, 1)}
    return result


def load_features(path, role, device):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["test_data_used"] or metadata["split"] != role or metadata["feature_schema"] != SCHEMA or metadata["full_vocabulary"] != 60000:
        raise ValueError("feature split, schema, or vocabulary differs")
    if metadata["feature_file_sha256"] != digest(path):
        raise ValueError("feature file hash differs")
    with np.load(path, allow_pickle=False) as saved:
        data = {key: torch.tensor(saved[key], device=device) for key in ["features", "candidates", "labels", "difficulty"]}
    raw = data["features"]
    if raw.shape != (metadata["samples"], 128, 20) or data["candidates"].shape != raw.shape[:2]:
        raise ValueError("incomplete query or candidate dimensions")
    if data["labels"].shape != raw.shape[:1] or data["difficulty"].shape != raw.shape[:1]:
        raise ValueError("incomplete label or reporting dimensions")
    if torch.any((data["candidates"] < 0) | (data["candidates"] >= 60000)) or torch.any((data["labels"] < 0) | (data["labels"] >= 60000)):
        raise ValueError("candidate or target outside the complete vocabulary")
    if torch.any((data["difficulty"] < 0) | (data["difficulty"] > 2)):
        raise ValueError("invalid reporting difficulty")
    data["features"] = derived_features(raw)
    return data, metadata


@torch.no_grad()
def evaluate(model, data, mean, scale):
    corrections = torch.cat([model.layers(((part - mean) / scale).clamp(-12, 12)).squeeze(-1)
                             for part in data["features"].split(256)])
    baseline = -data["features"][..., 1] / 0.1
    trials = [{"blend": blend, "validation": metrics(baseline + blend * corrections, data["candidates"], data["labels"], data["difficulty"])}
              for blend in BLENDS]
    # Zero correction wins ties, preserving the recorded baseline.
    return max(trials, key=lambda trial: trial["validation"]["memory"]["correct_top1"]), trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=900701)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1:
        parser.error("epochs and batch size must be positive")
    if args.output.exists():
        raise ValueError("training output exists; use a new stage name")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    train, train_info = load_features(args.train, "train", args.device)
    development, dev_info = load_features(args.development, "development", args.device)
    if train_info["native_artifacts_sha256"] != dev_info["native_artifacts_sha256"] or train_info["executable_snapshots_sha256"] != dev_info["executable_snapshots_sha256"]:
        raise ValueError("training and development feature implementations differ")
    if train_info["source_sha256"] != dev_info["source_sha256"] or train_info["dictionary_sha256"] != dev_info["dictionary_sha256"]:
        raise ValueError("training and development extraction sources differ")
    if model_inventory(args.model) != train_info["native_artifacts_sha256"]:
        raise ValueError("source native model differs from cached features")
    manifest = json.loads((args.model / "manifest.json").read_text())
    declared = manifest["training"]["validation_corpus"]
    validate_roles(train_info["corpora"], declared, "train", train_info["samples"])
    validate_roles(dev_info["corpora"], declared, "development", dev_info["samples"])
    raw = train["features"]
    mean, scale = raw.mean(dim=(0, 1)), raw.std(dim=(0, 1)).clamp_min(0.01)
    normalized = ((raw - mean) / scale).clamp(-12, 12)
    positives = train["candidates"] == train["labels"][:, None]
    eligible = positives.any(1).nonzero().flatten()
    targets = positives.float().argmax(1)
    if not len(eligible):
        raise ValueError("no naturally retrieved training positives")
    model = Ranker().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    baseline, baseline_trials = evaluate(model, development, mean, scale)
    for name in baseline["validation"]:
        if any(baseline["validation"][name][key] != dev_info["baseline"][name][key] for key in ["correct_top1", "correct_top5"]):
            raise ValueError("initialized correction changed native baseline hit counts")
    best = baseline["validation"]["memory"]["correct_top1"]
    args.output.mkdir(parents=True)
    with (args.model / "words.bin").open("rb") as source:
        bank_header = source.read(16)
    with (args.model / "spelling.bin").open("rb") as source:
        spelling_header = source.read(32)
    bindings = {"encoder_tag": struct.unpack_from("<Q", bank_header, 8)[0],
                "bank_tag": struct.unpack_from("<Q", spelling_header, 24)[0], "spelling_tag": manifest["spelling"]["tag"]}
    feature_names = FEATURES.copy()
    feature_names[2] = "geometry_margin"
    run = {"test_data_used": False, "train_metadata_sha256": digest(args.train.with_suffix(".json")),
           "development_metadata_sha256": digest(args.development.with_suffix(".json")),
           "base_manifest_sha256": train_info["base_manifest_sha256"], "bindings": bindings,
           "input_feature_schema": SCHEMA, "feature_schema": DERIVED_SCHEMA, "feature_names": feature_names,
           "eligible_training_queries": len(eligible), "training_queries": len(raw),
           "baseline": baseline, "baseline_trials": baseline_trials,
           "selection": "Exact hard-development top1 over every row; zero blend included; no alias credit",
           "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "source_sha256": {name: digest(ROOT / "training" / name) for name in ["train_complete_geometry.py", "train_reranker.py", "prepare_complete_geometry.py"]},
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "native_supported": False}
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    def save(epoch, selected):
        layers = [{"weights": layer.weight.detach().cpu().tolist(), "bias": layer.bias.detach().cpu().tolist()}
                  for layer in model.layers if isinstance(layer, torch.nn.Linear)]
        exported = {"schema_version": 1, "feature_schema": DERIVED_SCHEMA, **bindings,
                    "mean": mean.cpu().tolist(), "scale": scale.cpu().tolist(), "layers": layers,
                    "shortlist": 128, "residual_weight": -0.1 * selected["blend"], "trust_distance": 0.0}
        target = args.output / "geometry-ranker.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(exported) + "\n")
        temporary.replace(target)
        metadata = {"epoch": epoch, "selected": selected, "geometry_ranker_sha256": digest(target)}
        target = args.output / "best-metrics.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(target)
    save(0, baseline)
    print(json.dumps({"baseline": baseline, "eligible_training_queries": len(eligible)}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = eligible[torch.randperm(len(eligible), device=args.device)]
        loss_sum = 0.0
        for indices in order.split(args.batch):
            correction = model.layers(normalized[indices]).squeeze(-1)
            logits = -raw[indices, :, 1] / 0.1 + correction
            loss = F.cross_entropy(logits, targets[indices]) + 0.0001 * correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite geometry correction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        schedule.step()
        model.eval()
        selected, trials = evaluate(model, development, mean, scale)
        correct = selected["validation"]["memory"]["correct_top1"]
        improved = correct > best
        if improved:
            best = correct
            save(epoch + 1, selected)
        row = {"epoch": epoch + 1, "loss": loss_sum / len(eligible), "selected": selected, "trials": trials, "improved": improved}
        with (args.output / "history.jsonl").open("a") as target:
            target.write(json.dumps(row) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "trials"}), flush=True)


if __name__ == "__main__":
    main()
