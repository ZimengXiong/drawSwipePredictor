#!/usr/bin/env python3
"""Select complete word-and-character combinations on development writers only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from character_model import CharacterDecoder, character_targets
from retrieval_model import build_encoder, encoder_features
from train_character import encode, evaluate
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows, info = corpus(args.data)
    primary, primary_hash = load_checkpoint(args.primary)
    auxiliary, auxiliary_hash = load_checkpoint(args.auxiliary)
    if primary["words"] != auxiliary["words"]:
        raise ValueError("complete-system vocabularies differ")
    for saved in [primary, auxiliary]:
        if saved["metadata"]["validation_corpus"]["sha256"] != info["sha256"]:
            raise ValueError("complete-system selection requires the declared development corpus")
    if "character_model" not in primary:
        raise ValueError("primary checkpoint must contain its joint character decoder")
    models = []
    for saved in [primary, auxiliary]:
        architecture = saved["metadata"]["arguments"].get("architecture", "small")
        encoder = build_encoder(architecture).to("mps").eval()
        encoder.load_state_dict(saved["model"])
        encoder.requires_grad_(False)
        models.append((encoder, saved["proxies"].to("mps")))
    decoder = CharacterDecoder(getattr(models[0][0], "input_channels", 8)).to("mps").eval()
    decoder.load_state_dict(primary["character_model"])
    targets, lengths, _ = character_targets(primary["words"])
    paths = torch.tensor(np.array(rows["points"]), device="mps")
    labels = torch.tensor(np.array(rows["label"], dtype=np.int64), device="mps")
    difficulty = torch.tensor(np.array(rows["difficulty"], dtype=np.int64), device="mps")
    weights = [0.0, 0.25, 0.5, 0.75]
    cached = {name: [] for name in ["sequence", "raw", "embedding"]}
    pools = {weight: {"similarity": [], "candidates": []} for weight in weights}
    with torch.no_grad():
        for part in paths.split(192):
            sequence, raw, embedding = encode(models[0][0], part)
            for name, values in zip(cached, [sequence, raw, embedding]):
                cached[name].append(values)
            first = proxy_scores(embedding, models[0][1])
            other = proxy_scores(models[1][0](encoder_features(models[1][0], part)), models[1][1])
            for weight in weights:
                scores, candidates = ((1 - weight) * first + weight * other).topk(128, dim=1)
                pools[weight]["similarity"].append(scores)
                pools[weight]["candidates"].append(candidates)
        validation = {name: torch.cat(values) for name, values in cached.items()}
        validation.update(labels=labels, difficulty=difficulty)
        trials = []
        for weight in weights:
            validation.update({name: torch.cat(values) for name, values in pools[weight].items()})
            selected, calibration = evaluate(decoder, validation, targets, lengths)
            recall = {}
            for i, name in enumerate(["clean", "loose", "memory"]):
                mask = difficulty == i
                present = (validation["candidates"][mask] == labels[mask, None]).any(1)
                recall[name] = float(present.float().mean())
            trial = {"auxiliary_weight": weight, **selected, "shortlist_recall": recall, "calibration": calibration}
            trials.append(trial)
            print(json.dumps({key: value for key, value in trial.items() if key != "calibration"}), flush=True)
    report = {"primary_sha256": primary_hash, "auxiliary_sha256": auxiliary_hash,
        "primary_path": str(args.primary), "auxiliary_path": str(args.auxiliary), "development": info,
        "test_data_used": False, "full_vocabulary": len(primary["words"]), "trials": trials,
        "selected": max(trials, key=lambda row: row["validation"]["memory"]["top1"]),
        "rules": "All natural top128 candidates; exact-word metrics; no target insertion or alias credit"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
