#!/usr/bin/env python3
"""Compare hard training-pool and development accuracy without changing models."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from retrieval_model import build_encoder
from train_joint import load_checkpoint
from train_proxy import corpus, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/memory-proxy/train.bin"))
    parser.add_argument("--extra-train", type=Path, action="append", default=[])
    parser.add_argument("--validation", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=900491)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("sample count must be positive")
    torch.set_num_threads(4)
    loaded = [load_checkpoint(path) for path in args.checkpoint]
    validation, val_info = corpus(args.validation)
    declared = set()
    for saved, _ in loaded:
        if saved["metadata"]["validation_corpus"]["sha256"] != val_info["sha256"] or saved["metadata"]["test_data_used"]:
            raise ValueError("diagnostic requires the declared development corpus")
        if saved["words"] != loaded[0][0]["words"] or "proxies" not in saved or "character_model" in saved or saved["proxies"].shape[0] != len(saved["words"]):
            raise ValueError("diagnostic requires aligned word-proxy checkpoints")
        declared.update(info["sha256"] for info in [saved["metadata"]["train_corpus"], *saved["metadata"].get("extra_train_corpora", [])])
    datasets, descriptions = {}, {}
    for i, path in enumerate([args.train, *args.extra_train]):
        rows, info = corpus(path)
        if info["sha256"] not in declared or info["vocabulary"] != len(loaded[0][0]["words"]):
            raise ValueError("diagnostic pool is not declared training data")
        if not (info["writer_end"] <= val_info["writer_start"] or val_info["writer_end"] <= info["writer_start"]):
            raise ValueError("diagnostic training and development writers overlap")
        eligible = np.flatnonzero(rows["difficulty"] == 2)
        if len(eligible) < args.samples:
            raise ValueError("diagnostic pool has too few hard drawings")
        selected = np.random.default_rng(args.seed + i).choice(eligible, args.samples, replace=False)
        key = f"training_pool_{i}"
        datasets[key] = (torch.tensor(np.array(rows["points"][selected]), device="mps"),
                         torch.tensor(np.array(rows["label"][selected], dtype=np.int64), device="mps"))
        descriptions[key] = {"path": str(path), **info, "evaluated_hard_rows": args.samples}
    mask = validation["difficulty"] == 2
    datasets["development_memory"] = (torch.tensor(np.array(validation["points"][mask]), device="mps"),
                                      torch.tensor(np.array(validation["label"][mask], dtype=np.int64), device="mps"))
    reports = []
    for path, (saved, source_hash) in zip(args.checkpoint, loaded):
        model = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
        model.load_state_dict(saved["model"])
        model.requires_grad_(False)
        metrics = evaluate(model, saved["proxies"].to("mps"), datasets)
        own = {record["sha256"] for record in [saved["metadata"]["train_corpus"], *saved["metadata"].get("extra_train_corpora", [])]}
        entry = {"checkpoint": str(path), "sha256": source_hash, "epoch": saved["epoch"] + 1,
                 "declared_training_pools": {name: description["sha256"] in own for name, description in descriptions.items()},
                 "metrics": metrics}
        reports.append(entry)
        print(json.dumps(entry), flush=True)
    report = {"test_data_used": False, "full_vocabulary": len(loaded[0][0]["words"]), "development": val_info,
              "training_pools": descriptions, "seed": args.seed, "checkpoints": reports,
              "rules": "Models remain in evaluation mode; no weights or normalization updates; all word scores eligible; labels and writer IDs excluded from encoder inputs",
              "limits": "Training-pool samples may have been seen during fitting. Only declared development accuracy supports model selection. No benchmark or blind-confirmation evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
