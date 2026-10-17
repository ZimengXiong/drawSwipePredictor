#!/usr/bin/env python3
"""Train character evidence and lexical ranking on fresh development-safe data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from character_model import CharacterDecoder, character_targets, candidate_ctc_cost
from motor_sampler import MotorSampler
from retrieval_model import build_encoder, encoder_features, load_recordings
from train_proxy import corpus, proxy_scores

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def encode(encoder, paths):
    raw = encoder_features(encoder, paths)
    sequence = encoder.encode_sequence(raw)
    return sequence, raw, encoder.embed_sequence(sequence)


@torch.no_grad()
def evaluate(decoder, validation, targets, lengths):
    costs = []
    for start in range(0, len(validation["labels"]), 24):
        logits = decoder(validation["sequence"][start:start + 24], validation["raw"][start:start + 24], validation["embedding"][start:start + 24])
        costs.append(candidate_ctc_cost(logits.log_softmax(2), validation["candidates"][start:start + 24], targets, lengths))
    costs = torch.cat(costs)
    candidate_lengths = lengths[validation["candidates"].cpu()].to(costs.device)
    costs = costs / candidate_lengths.float().pow(0.3)
    best, trials = None, []
    for weight in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0]:
        logits = -costs + weight * 32 * validation["similarity"]
        ranked = validation["candidates"].gather(1, logits.topk(5, dim=1).indices)
        hits = ranked == validation["labels"][:, None]
        metrics = {}
        for i, name in enumerate(["clean", "loose", "memory"]):
            mask = validation["difficulty"] == i
            metrics[name] = {"samples": int(mask.sum()), "top1": float(hits[mask, 0].float().mean()), "top5": float(hits[mask].any(1).float().mean())}
        trial = {"proxy_weight": weight, "validation": metrics}
        trials.append(trial)
        if best is None or metrics["memory"]["top1"] > best["validation"]["memory"]["top1"]:
            best = trial
    return best, trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=900343)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["large", "intrinsic"]:
        raise ValueError("character decoder requires a residual backbone")
    device = "mps"
    encoder = build_encoder(architecture).to(device).eval()
    encoder.load_state_dict(saved["model"])
    encoder.requires_grad_(False)
    proxies = saved["proxies"].to(device)
    words = saved["words"]
    targets, lengths, aliases = character_targets(words)
    alias_gpu = aliases.to(device)
    sampler = MotorSampler(words, device)
    source = ROOT / "training/data/memory-proxy"
    train_rows, train_info = corpus(source / "train.bin")
    val_rows, val_info = corpus(source / "validation.bin")
    if train_info["sha256"] != saved["metadata"]["train_corpus"]["sha256"] or val_info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("character fitting must preserve the encoder's declared data splits")
    train_paths = torch.tensor(np.array(train_rows["points"]), device=device)
    train_labels = torch.tensor(np.array(train_rows["label"], dtype=np.int64), device=device)
    chooser = np.random.default_rng(args.seed)
    selected = chooser.choice(len(val_rows), 3000, replace=False)
    val_paths = torch.tensor(np.array(val_rows["points"][selected]), device=device)
    validation = {"labels": torch.tensor(np.array(val_rows["label"][selected], dtype=np.int64), device=device),
                  "difficulty": torch.tensor(np.array(val_rows["difficulty"][selected]), device=device)}
    with torch.no_grad():
        sequence, raw, embeddings = [], [], []
        for part in val_paths.split(256):
            seq, inp, vector = encode(encoder, part)
            sequence.append(seq); raw.append(inp); embeddings.append(vector)
        validation.update(sequence=torch.cat(sequence), raw=torch.cat(raw), embedding=torch.cat(embeddings))
        similarity = proxy_scores(validation["embedding"], proxies)
        values, candidates = similarity.topk(128, dim=1)
        validation.update(candidates=candidates, similarity=values)
    ids = {word: i for i, word in enumerate(words)}
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, ids)
    real, real_labels = torch.tensor(real, device=device), torch.tensor(real_labels, device=device)
    decoder = CharacterDecoder(getattr(encoder, "input_channels", 8)).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.0007, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00002)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"encoder_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "encoder_path": str(args.checkpoint),
        "train": train_info, "validation": val_info, "seed": args.seed, "test_data_used": False,
        "selection": "Memory development top1 over unmodified naturally retrieved top128 candidates",
        "targets": "Collapsed character sequences for CTC only; final metrics require exact full words",
        "source": "CTC and ranking supervision; no evaluation target is inserted into a shortlist"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best = -1.0
    for epoch in range(args.epochs):
        began = time.monotonic()
        decoder.train()
        accumulated = 0.0
        for step in range(args.steps):
            nr = args.batch * 3 // 10
            ns = args.batch * 2 // 10
            nf = args.batch - nr - ns
            ri = torch.randint(len(real), (nr,), device=device)
            si = torch.randint(len(train_paths), (ns,), device=device)
            fresh_labels = sampler.labels(nf)
            paths = torch.cat([real[ri], train_paths[si], sampler.sample(fresh_labels)])
            labels = torch.cat([real_labels[ri], train_labels[si], fresh_labels])
            sequence, raw, embedding = encode(encoder, paths)
            log_probs = decoder(sequence, raw, embedding).log_softmax(2)
            positive = candidate_ctc_cost(log_probs, labels[:, None], targets, lengths)[:, 0]
            normalizer = lengths[labels.cpu()].to(device).float()
            loss = (positive / normalizer).mean()
            if epoch >= 5:
                with torch.no_grad():
                    similarity = proxy_scores(embedding, proxies)
                    similarity.masked_fill_(alias_gpu[None] == alias_gpu[labels, None], -torch.inf)
                    negatives = similarity.topk(7, dim=1).indices
                    candidates = torch.cat([labels[:, None], negatives], dim=1)
                    prior = proxy_scores(embedding, proxies).gather(1, candidates)
                cost = candidate_ctc_cost(log_probs, candidates, targets, lengths)
                scores = -cost / lengths[candidates.cpu()].to(device).float().pow(0.3) + 0.25 * 32 * prior
                loss = 0.5 * loss + F.cross_entropy(scores, torch.zeros(args.batch, dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite character loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
            optimizer.step()
            accumulated += loss.item()
        scheduler.step()
        decoder.eval()
        result, trials = evaluate(decoder, validation, targets, lengths)
        score = result["validation"]["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        report = {"epoch": epoch + 1, "loss": accumulated / args.steps, **result,
                  "best_score": best, "improved": improved, "seconds": time.monotonic() - began}
        print(json.dumps(report), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save({"model": decoder.state_dict(), "metadata": metadata, "epoch": epoch, "selected": result,
                        "raw_channels": getattr(encoder, "input_channels", 8)}, temporary)
            temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps({**report, "trials": trials}, indent=2) + "\n")


if __name__ == "__main__":
    main()
