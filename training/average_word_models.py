#!/usr/bin/env python3
"""Compare aligned model averages using training-only normalization updates."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import build_encoder, encoder_features
from train_joint import load_checkpoint
from train_proxy import corpus, evaluate, sample_corpora


def average_state(first, second, fraction):
    """Reject incompatible states and preserve integer tracking buffers."""
    if not 0 <= fraction <= 1 or first.keys() != second.keys():
        raise ValueError("incompatible state keys or averaging fraction")
    result = {}
    for name, value in first.items():
        other = second[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"incompatible state tensor: {name}")
        if value.is_floating_point():
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError("nonfinite model state")
            result[name] = value * (1 - fraction) + other * fraction
        else:
            result[name] = torch.maximum(value, other)
    return result


@torch.no_grad()
def refresh_normalization(model, shard, batches, batch, seed):
    modules = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    momenta = [module.momentum for module in modules]
    if batches:
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
        model.train()
        rng = np.random.default_rng(seed)
        for _ in range(batches):
            paths, _ = sample_corpora([shard], batch, rng, "mps")
            model(encoder_features(model, paths))
    model.eval()
    for module, momentum in zip(modules, momenta):
        module.momentum = momentum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/memory-proxy/train.bin"))
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--bn-batches", type=int, default=0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=900479)
    args = parser.parse_args()
    if args.bn_batches < 0 or args.batch < 1 or any(not 0 <= f <= 1 for f in args.fractions):
        parser.error("invalid batch count or averaging fraction")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    loaded = [load_checkpoint(path) for path in [args.first, args.second]]
    sources = [item[0] for item in loaded]
    architecture = sources[0]["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["large", "intrinsic", "giant", "wide512"]:
        raise ValueError("averaging requires aligned residual encoder architectures")
    train, train_info = corpus(args.train)
    rows, info = corpus(args.data)
    for saved in sources:
        if saved["words"] != sources[0]["words"] or saved["metadata"]["arguments"].get("architecture", "small") != architecture:
            raise ValueError("model vocabularies or architectures differ")
        if saved["metadata"]["validation_corpus"]["sha256"] != info["sha256"] or saved["metadata"]["train_corpus"]["sha256"] != train_info["sha256"] or saved["metadata"]["test_data_used"]:
            raise ValueError("averaging requires the declared training and development writers")
        if saved["proxies"].ndim != 2 or saved["proxies"].shape != sources[0]["proxies"].shape:
            raise ValueError("averaging requires one aligned word vector per class")
    datasets = {}
    for level, name in enumerate(["clean", "loose", "memory"]):
        mask = rows["difficulty"] == level
        datasets[name] = (torch.tensor(np.array(rows["points"][mask]), device="mps"),
                          torch.tensor(np.array(rows["label"][mask], dtype=np.int64), device="mps"))
    model = build_encoder(architecture).to("mps").eval()
    trials, best, best_saved, best_source = [], None, None, None
    options = [("original", float(i), i) for i in range(2)]
    method = "average_with_training_normalization" if args.bn_batches else "average"
    options += [(method, fraction, None) for fraction in args.fractions]
    for method, fraction, source in options:
        state = sources[source]["model"] if source is not None else average_state(sources[0]["model"], sources[1]["model"], fraction)
        model.load_state_dict(state)
        proxies = sources[source]["proxies"] if source is not None else F.normalize(
            sources[0]["proxies"] * (1 - fraction) + sources[1]["proxies"] * fraction, dim=1)
        proxies = proxies.to("mps")
        if source is None:
            refresh_normalization(model, (train, train_info), args.bn_batches, args.batch, args.seed)
        metrics = evaluate(model, proxies, datasets)
        trial = {"method": method, "second_fraction": fraction, "validation": metrics}
        trials.append(trial)
        print(json.dumps(trial), flush=True)
        if best is None or metrics["memory"]["top1"] > best["validation"]["memory"]["top1"]:
            best, best_source = trial, source
            best_saved = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                          "proxies": proxies.detach().cpu().clone()}
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [{"path": str(path), "sha256": item[1], "epoch": item[0]["epoch"] + 1}
               for path, item in zip([args.first, args.second], loaded)]
    report = {"sources": reports, "development": info, "normalization_training": train_info,
              "normalization_batches": args.bn_batches, "normalization_batch_size": args.batch, "seed": args.seed,
              "full_vocabulary": len(sources[0]["words"]), "test_data_used": False,
              "trials": trials, "selected": best,
              "rules": "Original controls included; all development rows and full dictionary; normalization sees training paths only; labels and writer IDs are not encoder inputs"}
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if best_source is not None:
        path = [args.first, args.second][best_source]
        (args.output / "best.pt").write_bytes(path.read_bytes())
    else:
        metadata = {"arguments": {"architecture": architecture}, "method": "Aligned parameter and word-vector averaging with training-only normalization updates",
                    "train_corpus": train_info, "validation_corpus": info, "test_data_used": False,
                    "sources": reports, "selected": best,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        torch.save({**best_saved, "epoch": -1, "words": sources[0]["words"], "counts": sources[0]["counts"],
                    "best_score": best["validation"]["memory"]["top1"], "metadata": metadata}, args.output / "best.pt")


if __name__ == "__main__":
    main()
