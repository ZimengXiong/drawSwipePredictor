#!/usr/bin/env python3
"""Learn full spelling likelihoods on independent training writers."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from autoregressive_model import SpellingDecoder, word_tokens
from motor_sampler import MotorSampler
from retrieval_model import build_encoder, load_recordings
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores, sample_corpora, real_word_groups, balanced_real_indices

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, validation, inputs, targets, lengths):
    costs = []
    for start in range(0, len(validation["labels"]), 4):
        part = {name: value[start:start + 4] for name, value in validation.items()}
        memory = model.encode_memory(part["sequence"], part["raw"], part["embedding"])
        costs.append(model.candidate_cost(memory, part["candidates"], inputs, targets))
    costs = torch.cat(costs)
    sizes = lengths[validation["candidates"]].float()
    trials = []
    for power in [0.0, 0.3, 0.7, 1.0]:
        for weight in [None, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            if weight is None and power:
                continue
            scores = validation["similarity"] if weight is None else -costs / sizes.pow(power) + 32 * weight * validation["similarity"]
            ordered = scores.argsort(dim=1, descending=True, stable=True)[:, :5]
            ranked = validation["candidates"].gather(1, ordered)
            hits = ranked == validation["labels"][:, None]
            metrics = {}
            for difficulty, name in enumerate(["clean", "loose", "memory"]):
                mask = validation["difficulty"] == difficulty
                metrics[name] = {"samples": int(mask.sum()), "top1": float(hits[mask, 0].float().mean()),
                                 "top5": float(hits[mask].any(1).float().mean())}
            trials.append({"length_power": power, "proxy_weight": weight, "validation": metrics})
    return max(trials, key=lambda row: row["validation"]["memory"]["top1"]), trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-train", type=Path, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=900419)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--init-decoder", type=Path)
    parser.add_argument("--ranking-candidates", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch, args.evaluate_every, args.validation_samples) < 1:
        parser.error("counts must be positive")
    if args.ranking_candidates and (not 2 <= args.ranking_candidates <= 128 or not args.init_decoder):
        parser.error("ranking requires an initialized decoder and 2 to 128 candidates")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps"
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["large", "intrinsic", "giant"]:
        raise ValueError("spelling decoder requires a residual encoder")
    encoder = build_encoder(architecture).to(device).eval().requires_grad_(False)
    encoder.load_state_dict(saved["model"])
    proxies, words = saved["proxies"].to(device), saved["words"]
    inputs, targets, lengths = [values.to(device) for values in word_tokens(words)]
    if inputs.shape[1] > 64:
        raise ValueError("word tokens exceed decoder position capacity")
    source = ROOT / "training/data/memory-proxy"
    shards = [corpus(source / "train.bin")] + [corpus(path) for path in args.extra_train]
    val_rows, val_info = corpus(source / "validation.bin")
    if shards[0][1]["sha256"] != saved["metadata"]["train_corpus"]["sha256"] or val_info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("spelling training must preserve the declared encoder splits")
    declared = [info for _, info in shards] + [val_info]
    for i, info in enumerate(declared):
        if info["seed"] in [101, 102, 9001, 12345, 246810, 700007] or info["vocabulary"] != len(words):
            raise ValueError("reserved writer block or mismatched vocabulary")
        for other in declared[:i]:
            if not (info["writer_end"] <= other["writer_start"] or other["writer_end"] <= info["writer_start"]):
                raise ValueError("training and development writer blocks overlap")
    if args.validation_samples > len(val_rows):
        parser.error("requested development sample count is unavailable")
    validation_seed = args.seed if args.validation_seed is None else args.validation_seed
    selected = np.random.default_rng(validation_seed).choice(len(val_rows), args.validation_samples, replace=False)
    paths = torch.tensor(np.array(val_rows["points"][selected]), device=device)
    validation = {"labels": torch.tensor(np.array(val_rows["label"][selected], dtype=np.int64), device=device),
                  "difficulty": torch.tensor(np.array(val_rows["difficulty"][selected], dtype=np.int64), device=device)}
    with torch.no_grad():
        encoded = [encode(encoder, part) for part in paths.split(192)]
        for i, name in enumerate(["sequence", "raw", "embedding"]):
            validation[name] = torch.cat([part[i] for part in encoded])
        ranked = [proxy_scores(part, proxies).topk(128, dim=1) for part in validation["embedding"].split(192)]
        validation.update(similarity=torch.cat([part.values for part in ranked]), candidates=torch.cat([part.indices for part in ranked]))
    del paths, encoded, ranked, saved
    sampler = MotorSampler(words, device)
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, {word: i for i, word in enumerate(words)})
    groups = real_word_groups(real_labels, device)
    real, real_labels = torch.tensor(real, device=device), torch.tensor(real_labels, device=device)
    model = SpellingDecoder(getattr(encoder, "input_channels", 8)).to(device)
    decoder_hash = None
    if args.init_decoder:
        initial, decoder_hash = load_checkpoint(args.init_decoder)
        if initial["metadata"]["encoder_sha256"] != encoder_hash or initial["words"] != words:
            raise ValueError("decoder initialization belongs to a different encoder or dictionary")
        model.load_state_dict(initial["model"])
        del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    metadata = {"encoder_sha256": encoder_hash, "encoder_path": str(args.checkpoint),
        "decoder_initialization_sha256": decoder_hash,
        "objective": "Listwise candidate classification plus 0.25 times letter cross entropy" if args.ranking_candidates else "Full-spelling teacher-forced cross entropy",
        "arguments": {key: [str(p) for p in value] if key == "extra_train" else str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_corpora": [info for _, info in shards], "validation_corpus": val_info,
        "real_training_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "real_sampling": "uniform observed training word types; all recordings retained",
        "source_sha256": {name: hashlib.sha256((ROOT / "training" / name).read_bytes()).hexdigest()
                          for name in ["autoregressive_model.py", "train_autoregressive.py", "motor_sampler.py"]},
        "test_data_used": False, "selection": "Exact-word development top1; natural top128 from full vocabulary; no target injection",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    baseline = validation["candidates"][:, 0] == validation["labels"]
    memory_mask = validation["difficulty"] == 2
    best = float(baseline[memory_mask].float().mean())
    print(json.dumps({"initial_retrieval_memory_top1": best, "samples": int(memory_mask.sum())}), flush=True)
    if args.init_decoder:
        model.eval()
        chosen, trials = evaluate(model, validation, inputs, targets, lengths)
        best = chosen["validation"]["memory"]["top1"]
        initial_report = {"epoch": 0, "selected": chosen, "trials": trials, "best_score": best}
        print(json.dumps({"initial_spelling_validation": chosen}), flush=True)
        initial_path = args.output / "best.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                    "epoch": -1, "best_score": best, "selected": chosen, "metadata": metadata, "words": words}, initial_path)
        initial_path.replace(args.output / "best.pt")
        (args.output / "best-metrics.json").write_text(json.dumps(initial_report, indent=2) + "\n")
    for epoch in range(args.epochs):
        model.train()
        began, accumulated = time.monotonic(), 0.0
        for step in range(args.steps):
            nr, ns = args.batch * 15 // 100, args.batch // 4
            nf = args.batch - nr - ns
            ri = balanced_real_indices(groups, nr)
            synthetic_paths, synthetic_labels = sample_corpora(shards, ns, rng, device)
            fresh = sampler.labels(nf)
            paths = torch.cat([real[ri], synthetic_paths, sampler.sample(fresh)])
            labels = torch.cat([real_labels[ri], synthetic_labels, fresh])
            with torch.no_grad():
                sequence, raw, embeddings = encode(encoder, paths)
            memory = model.encode_memory(sequence, raw, embeddings)
            logits = model(memory, inputs[labels])
            loss = F.cross_entropy(logits.flatten(0, 1), targets[labels].flatten(), ignore_index=0)
            if args.ranking_candidates:
                with torch.no_grad():
                    similarity = proxy_scores(embeddings, proxies)
                    alternatives = similarity.masked_fill(torch.arange(len(words), device=device)[None] == labels[:, None], -torch.inf).topk(128, dim=1).indices
                    priority = torch.rand(alternatives.shape, device=device)
                    priority[:, :16] += 1
                    columns = priority.topk(args.ranking_candidates - 1, dim=1).indices
                    # Labeled positives are training supervision only. The
                    # development evaluator always retrieves its own candidates.
                    candidates = torch.cat([labels[:, None], alternatives.gather(1, columns)], dim=1)
                    prior = similarity.gather(1, candidates)
                cost = model.candidate_cost(memory, candidates, inputs, targets)
                scores = -cost / lengths[candidates].float().pow(0.3) + 32 * prior
                ranking = F.cross_entropy(scores, torch.zeros(len(paths), dtype=torch.long, device=device))
                loss = ranking + 0.25 * loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite spelling loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            accumulated += loss.item()
        schedule.step()
        report = {"epoch": epoch + 1, "loss": accumulated / args.steps, "seconds": time.monotonic() - began}
        chosen, improved = None, False
        if epoch == 0 or (epoch + 1) % args.evaluate_every == 0:
            model.eval()
            chosen, trials = evaluate(model, validation, inputs, targets, lengths)
            score = chosen["validation"]["memory"]["top1"]
            improved = score > best
            best = max(best, score)
            report.update(selected=chosen, trials=trials, best_score=best, improved=improved)
        print(json.dumps({key: value for key, value in report.items() if key != "trials"}), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                      "epoch": epoch, "best_score": best, "selected": chosen, "metadata": metadata, "words": words}
        temporary = args.output / "latest.pt.tmp"
        torch.save(checkpoint, temporary); temporary.replace(args.output / "latest.pt")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save(checkpoint, temporary); temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
