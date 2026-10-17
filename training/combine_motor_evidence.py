#!/usr/bin/env python3
"""Check whether motor fits complement fixed word-and-character evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from character_edit import ctc_edit_components, edit_log_likelihood
from character_model import CharacterDecoder, character_targets, candidate_ctc_cost
from retrieval_model import build_encoder, encoder_features
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["primary", "auxiliary", "fit_retriever", "fits", "motor_calibration", "complete_selection", "output"]:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--fixed-combination", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    primary, primary_hash = load_checkpoint(args.primary)
    auxiliary, auxiliary_hash = load_checkpoint(args.auxiliary)
    old, old_hash = load_checkpoint(args.fit_retriever)
    words = primary["words"]
    if len(words) != 60000 or words != auxiliary["words"] or words != old["words"]:
        raise ValueError("motor combination requires matching full 60000-word banks")
    del old
    fits = torch.load(args.fits, map_location="cpu", weights_only=False)
    fit_meta = fits["metadata"]
    complete = json.loads(args.complete_selection.read_text())
    calibration = json.loads(args.motor_calibration.read_text())
    rows, info = corpus(args.data)
    if any(record["test_data_used"] for record in [complete, calibration, fit_meta]):
        raise ValueError("motor combination must use development data only")
    if complete["primary_sha256"] != primary_hash or complete["auxiliary_sha256"] != auxiliary_hash:
        raise ValueError("complete-system source checkpoints differ")
    if fit_meta["checkpoint_sha256"] != old_hash or calibration["frozen_retriever_sha256"] != old_hash:
        raise ValueError("motor-fit source checkpoint differs")
    declared = [complete["development"]["sha256"], calibration["development_sha256"], fit_meta["development_sha256"],
        primary["metadata"]["validation_corpus"]["sha256"], auxiliary["metadata"]["validation_corpus"]["sha256"]]
    if any(value != info["sha256"] for value in declared):
        raise ValueError("motor combination requires the declared development corpus")
    offset, count = fit_meta["arguments"].get("offset", 0), fit_meta["samples"]
    selected = np.flatnonzero(rows["difficulty"] == 2)[offset:offset + count]
    labels = torch.tensor(np.array(rows["label"][selected], dtype=np.int64))
    if not torch.equal(labels, fits["labels"]):
        raise ValueError("motor-fit rows are not aligned with development geometry")
    fixed = json.loads(args.fixed_combination.read_text()) if args.fixed_combination else None
    if fixed:
        for name, value in [("primary_sha256", primary_hash), ("auxiliary_sha256", auxiliary_hash), ("fit_retriever_sha256", old_hash)]:
            if fixed[name] != value:
                raise ValueError("fixed combination source checkpoints differ")
        for key, path in [("complete_selection_sha256", args.complete_selection), ("motor_calibration_sha256", args.motor_calibration)]:
            if fixed[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("fixed combination calibration sources differ")
        earlier = fixed["writer_selection"]
        if not (offset >= earlier["offset"] + earlier["samples"] or offset + count <= earlier["offset"]):
            raise ValueError("fixed-combination development slices overlap")
    settings = complete["selected"]
    if settings["proxy_weight"] is None:
        raise ValueError("this combination expects the selected character evidence")
    models = []
    for saved in [primary, auxiliary]:
        encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
        encoder.load_state_dict(saved["model"]); encoder.requires_grad_(False)
        models.append((encoder, saved["proxies"].to("mps")))
    decoder = CharacterDecoder(getattr(models[0][0], "input_channels", 8)).to("mps").eval()
    decoder.load_state_dict(primary["character_model"])
    targets, lengths, _ = character_targets(words)
    paths = torch.tensor(np.array(rows["points"][selected]), device="mps")
    motor = (fits["errors"] + calibration["selected"]["regularization"] * fits["penalties"]).min(2).values
    if not torch.isfinite(motor).all():
        raise ValueError("nonfinite motor fits; do not omit development rows")
    middle = motor.median(1).values
    evidence = (motor - middle[:, None]) / (middle - motor.min(1).values).clamp_min(1e-6)[:, None]
    candidates_all, valid_all, costs_all, motor_all = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(paths), 24):
            part = paths[start:start + 24]
            sequence, raw, embedding = encode(models[0][0], part)
            weight = settings["auxiliary_weight"]
            first = proxy_scores(embedding, models[0][1])
            second = proxy_scores(models[1][0](encoder_features(models[1][0], part)), models[1][1])
            prior = (1 - weight) * first + weight * second
            natural = prior.topk(128, dim=1).indices.cpu().tolist()
            pool = torch.zeros(len(part), 192, dtype=torch.long)
            valid = torch.zeros_like(pool, dtype=torch.bool)
            motor_evidence = torch.zeros_like(pool, dtype=torch.float32)
            for i, initial in enumerate(natural):
                positions = {word: j for j, word in enumerate(initial)}
                union = list(initial)
                for column, word in enumerate(fits["candidates"][start + i].tolist()):
                    if word not in positions:
                        positions[word] = len(union); union.append(word)
                    motor_evidence[i, positions[word]] = evidence[start + i, column]
                pool[i, :len(union)] = torch.tensor(union)
                valid[i, :len(union)] = True
            gpu_pool = pool.to("mps")
            probabilities = decoder(sequence, raw, embedding).log_softmax(2)
            sizes = lengths[pool].to("mps")
            probability = settings["deletion_probability"]
            if probability:
                ordinary, deleted = ctc_edit_components(probabilities, gpu_pool, targets, lengths)
                likelihood = edit_log_likelihood(ordinary, deleted, sizes, probability)
            else:
                likelihood = -candidate_ctc_cost(probabilities, gpu_pool, targets, lengths)
            scores = likelihood / sizes.float().pow(0.3) + settings["proxy_weight"] * 32 * prior.gather(1, gpu_pool)
            if not torch.isfinite(scores.cpu()[valid]).all():
                raise ValueError("nonfinite complete-system evidence")
            candidates_all.append(pool); valid_all.append(valid)
            costs_all.append(scores.cpu().masked_fill(~valid, -torch.inf)); motor_all.append(motor_evidence)
    candidates, valid, scores, evidence = [torch.cat(parts) for parts in [candidates_all, valid_all, costs_all, motor_all]]
    before = candidates[:, :128].gather(1, scores[:, :128].argmax(1)[:, None])[:, 0] == labels
    trials = []
    for weight in ([fixed["selected"]["motor_weight"]] if fixed else [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]):
        ranking = (scores - weight * evidence).argsort(dim=1, descending=True, stable=True)[:, :5]
        hits = candidates.gather(1, ranking) == labels[:, None]
        trials.append({"motor_weight": weight, "top1": float(hits[:, 0].float().mean()), "top5": float(hits.any(1).float().mean()),
            "gained": int((hits[:, 0] & ~before).sum()), "lost": int((~hits[:, 0] & before).sum())})
    report = {"primary_sha256": primary_hash, "auxiliary_sha256": auxiliary_hash, "fit_retriever_sha256": old_hash,
        "development_sha256": info["sha256"], "test_data_used": False, "full_vocabulary": len(words),
        "writer_selection": {"difficulty": "memory", "offset": offset, "samples": len(paths)},
        "baseline_complete_top1": float(before.float().mean()), "candidate_union_recall": float(((candidates == labels[:, None]) & valid).any(1).float().mean()),
        "trials": trials, "selected": max(trials, key=lambda row: row["top1"]), "calibration_frozen": fixed is not None,
        "complete_selection_sha256": hashlib.sha256(args.complete_selection.read_bytes()).hexdigest(),
        "motor_calibration_sha256": hashlib.sha256(args.motor_calibration.read_bytes()).hexdigest(),
        "rules": "Union of naturally retrieved lists; absent motor evidence is neutral; no target insertion or alias credit"}
    torch.save({"candidates": candidates, "valid": valid, "scores": scores, "motor_evidence": evidence,
        "labels": labels, "metadata": report}, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
