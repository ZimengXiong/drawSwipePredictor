#!/usr/bin/env python3
"""Jointly learn word retrieval and character alignment on training writers."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from character_model import CharacterDecoder, character_targets, candidate_ctc_cost
from motor_sampler import MotorSampler
from retrieval_model import build_encoder, encoder_features, load_recordings
from train_character import encode, evaluate
from train_proxy import corpus, proxy_scores

ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    content = path.read_bytes()
    return torch.load(io.BytesIO(content), map_location="cpu", weights_only=False), hashlib.sha256(content).hexdigest()


@torch.no_grad()
def refresh_validation(encoder, proxies, paths, labels, difficulty):
    encoded = [encode(encoder, part) for part in paths.split(192)]
    validation = {"labels": labels, "difficulty": difficulty}
    for i, name in enumerate(["sequence", "raw", "embedding"]):
        validation[name] = torch.cat([part[i] for part in encoded])
    ranked = [proxy_scores(part, proxies).topk(128, dim=1) for part in validation["embedding"].split(192)]
    validation.update(similarity=torch.cat([r.values for r in ranked]), candidates=torch.cat([r.indices for r in ranked]))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=900343)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = "mps"
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    initial_decoder, decoder_hash = load_checkpoint(args.decoder)
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["large", "intrinsic", "giant"]:
        raise ValueError("joint training requires a residual backbone")
    encoder = build_encoder(architecture).to(device)
    encoder.load_state_dict(saved["model"])
    channels = getattr(encoder, "input_channels", 8)
    if initial_decoder["raw_channels"] != channels:
        raise ValueError("transferred decoder input channels differ")
    decoder = CharacterDecoder(channels).to(device)
    decoder.load_state_dict(initial_decoder.get("character_model", initial_decoder["model"]))
    proxies = torch.nn.Parameter(saved["proxies"].to(device))
    words, counts = saved["words"], saved["counts"]
    targets, lengths, aliases = character_targets(words)
    aliases = aliases.to(device)
    sampler = MotorSampler(words, device)
    source = ROOT / "training/data/memory-proxy"
    train_rows, train_info = corpus(source / "train.bin")
    val_rows, val_info = corpus(source / "validation.bin")
    for name, info in [("train", train_info), ("validation", val_info)]:
        if info["sha256"] != saved["metadata"][name + "_corpus"]["sha256"]:
            raise ValueError("joint fitting must preserve the encoder's declared data splits")
    train_paths = torch.tensor(np.array(train_rows["points"]), device=device)
    train_labels = torch.tensor(np.array(train_rows["label"], dtype=np.int64), device=device)
    selected = np.random.default_rng(args.seed).choice(len(val_rows), 3000, replace=False)
    val_paths = torch.tensor(np.array(val_rows["points"][selected]), device=device)
    val_labels = torch.tensor(np.array(val_rows["label"][selected], dtype=np.int64), device=device)
    val_difficulty = torch.tensor(np.array(val_rows["difficulty"][selected], dtype=np.int64), device=device)
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, {word: i for i, word in enumerate(words)})
    real, real_labels = torch.tensor(real, device=device), torch.tensor(real_labels, device=device)
    optimizer = torch.optim.AdamW([{"params": encoder.parameters(), "lr": 0.00003},
        {"params": decoder.parameters(), "lr": 0.0002}, {"params": [proxies], "lr": 0.0001}], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.000005)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_corpus": train_info, "validation_corpus": val_info, "test_data_used": False,
        "initial_encoder_sha256": encoder_hash, "initial_decoder_sha256": decoder_hash,
        "initial_encoder_training": saved["metadata"], "initial_decoder_training": initial_decoder["metadata"],
        "real_training_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "selection": "Exact-word hard development top1 of naturally retrieved top128; jointly selected character/proxy mixing",
        "training_mix": {"real": 0.2, "cached_synthetic": 0.1, "fresh_synthetic": 0.7}}
    metadata["arguments"]["architecture"] = architecture
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    encoder.eval(); decoder.eval()
    validation = refresh_validation(encoder, proxies, val_paths, val_labels, val_difficulty)
    initial, _ = evaluate(decoder, validation, targets, lengths)
    print(json.dumps({"initial_validation": initial}), flush=True)
    best = initial["validation"]["memory"]["top1"]
    for epoch in range(args.epochs):
        encoder.train(); decoder.train()
        began, accumulated = time.monotonic(), 0.0
        for step in range(args.steps):
            nr, ns = args.batch // 5, args.batch // 10
            nf = args.batch - nr - ns
            ri = torch.randint(len(real), (nr,), device=device)
            si = torch.randint(len(train_paths), (ns,), device=device)
            fresh = sampler.labels(nf)
            paths = torch.cat([real[ri], train_paths[si], sampler.sample(fresh)])
            labels = torch.cat([real_labels[ri], train_labels[si], fresh])
            raw = encoder_features(encoder, paths)
            sequence = encoder.encode_sequence(raw)
            embedding = encoder.embed_sequence(sequence)
            similarity = proxy_scores(embedding, proxies)
            log_probs = decoder(sequence, raw, embedding).log_softmax(2)
            positive = candidate_ctc_cost(log_probs, labels[:, None], targets, lengths)[:, 0]
            positive = (positive / lengths[labels.cpu()].to(device).float()).mean()
            with torch.no_grad():
                filtered = similarity.masked_fill(aliases[None] == aliases[labels, None], -torch.inf)
                candidates = torch.cat([labels[:, None], filtered.topk(7, dim=1).indices], dim=1)
            likelihood = -candidate_ctc_cost(log_probs, candidates, targets, lengths)
            scores = likelihood / lengths[candidates.cpu()].to(device).float().pow(0.3) + 0.25 * 32 * similarity.gather(1, candidates)
            ranking = F.cross_entropy(scores, torch.zeros(args.batch, dtype=torch.long, device=device))
            loss = ranking + 0.25 * positive + 0.5 * F.cross_entropy(32 * similarity, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters(), proxies], 5)
            optimizer.step()
            accumulated += loss.item()
        scheduler.step()
        encoder.eval(); decoder.eval()
        validation = refresh_validation(encoder, proxies, val_paths, val_labels, val_difficulty)
        result, trials = evaluate(decoder, validation, targets, lengths)
        score = result["validation"]["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        report = {"epoch": epoch + 1, "loss": accumulated / args.steps, **result, "best_score": best,
            "improved": improved, "seconds": time.monotonic() - began}
        print(json.dumps(report), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save({"model": encoder.state_dict(), "character_model": decoder.state_dict(),
                "proxies": F.normalize(proxies.detach(), dim=-1), "words": words, "counts": counts,
                "metadata": metadata, "epoch": epoch, "best_score": best, "selected": result, "raw_channels": channels,
                "retriever_validation": next(trial["validation"] for trial in trials if trial["proxy_weight"] is None)}, temporary)
            temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps({**report, "trials": trials}, indent=2) + "\n")


if __name__ == "__main__":
    main()
