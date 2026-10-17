#!/usr/bin/env python3
"""Learn full spelling likelihoods on independent training writers."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from autoregressive_model import SpellingDecoder, word_tokens, decoder_dimensions, expand_decoder
from motor_sampler import MotorSampler
from retrieval_model import build_encoder, encoder_features, load_recordings
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores, sample_corpora, real_word_groups, balanced_real_indices

ROOT = Path(__file__).resolve().parents[1]


def validate_retrieval_source(retrieved, context):
    """Keep hard-negative mining inside the declared vocabulary and writer splits."""
    if retrieved["words"] != context["words"] or retrieved["metadata"]["test_data_used"]:
        raise ValueError("independent retriever vocabulary or data provenance differs")
    for key in ["train_corpus", "validation_corpus"]:
        if retrieved["metadata"][key]["sha256"] != context["metadata"][key]["sha256"]:
            raise ValueError("independent retriever must use the declared training and development writers")


def differentiable_context(encoder, paths):
    """Retain gradients through stroke features used by the spelling decoder."""
    raw = encoder_features(encoder, paths)
    sequence = encoder.encode_sequence(raw)
    return sequence, raw, encoder.embed_sequence(sequence)


@torch.no_grad()
def refresh_context(encoder, paths, validation):
    parts = [encode(encoder, part) for part in paths.split(192)]
    for i, name in enumerate(["sequence", "raw", "embedding"]):
        validation[name] = torch.cat([part[i] for part in parts])


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
    parser.add_argument("--retrieval-checkpoint", type=Path,
                        help="Independent frozen word model for mining negatives and development candidates")
    parser.add_argument("--decoder-layers", type=int, choices=[3, 6, 9, 12],
                        help="Decoder depth; defaults to the source depth or three blocks")
    parser.add_argument("--ranking-candidates", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--fine-tune-encoder", action="store_true")
    parser.add_argument("--encoder-learning-rate", type=float, default=0.00001)
    parser.add_argument("--fresh-fraction", type=float)
    parser.add_argument("--real-train", type=Path, default=ROOT / "training/data/futo-large/train.jsonl")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch, args.evaluate_every, args.validation_samples) < 1:
        parser.error("counts must be positive")
    if args.ranking_candidates and (not 2 <= args.ranking_candidates <= 128 or not args.init_decoder):
        parser.error("ranking requires an initialized decoder and 2 to 128 candidates")
    if args.fine_tune_encoder and not args.ranking_candidates:
        parser.error("encoder fine-tuning requires initialized candidate ranking")
    if args.retrieval_checkpoint and not args.ranking_candidates:
        parser.error("independent retrieval requires initialized candidate ranking")
    if args.fresh_fraction is not None and not 0 <= args.fresh_fraction <= 0.7:
        parser.error("fresh fraction must be between zero and 0.7")
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
    retriever = copy.deepcopy(encoder) if args.fine_tune_encoder else encoder
    if args.fine_tune_encoder:
        encoder.requires_grad_(True)
    proxies, words = saved["proxies"].to(device), saved["words"]
    retrieval_hash = encoder_hash
    if args.retrieval_checkpoint:
        retrieved, retrieval_hash = load_checkpoint(args.retrieval_checkpoint)
        validate_retrieval_source(retrieved, saved)
        retriever = build_encoder(retrieved["metadata"]["arguments"].get("architecture", "small")).to(device).eval().requires_grad_(False)
        retriever.load_state_dict(retrieved["model"])
        proxies = retrieved["proxies"].to(device)
        del retrieved
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
        retrieval_embeddings = validation["embedding"]
        if args.retrieval_checkpoint:
            retrieval_embeddings = torch.cat([retriever(encoder_features(retriever, part)) for part in paths.split(192)])
        ranked = [proxy_scores(part, proxies).topk(128, dim=1) for part in retrieval_embeddings.split(192)]
        validation.update(similarity=torch.cat([part.values for part in ranked]), candidates=torch.cat([part.indices for part in ranked]))
    validation_paths = paths
    del encoded, ranked, saved
    sampler = MotorSampler(words, device) if args.fresh_fraction != 0 else None
    real_path = args.real_train
    with real_path.open("rb") as real_source:
        real_hash = hashlib.file_digest(real_source, "sha256").hexdigest()
    real, real_labels = load_recordings(real_path, {word: i for i, word in enumerate(words)})
    groups = real_word_groups(real_labels, device)
    real, real_labels = torch.tensor(real, device=device), torch.tensor(real_labels, device=device)
    initial, decoder_hash = load_checkpoint(args.init_decoder) if args.init_decoder else (None, None)
    args.decoder_layers = args.decoder_layers or (decoder_dimensions(initial["model"])[1] if initial else 3)
    model = SpellingDecoder(getattr(encoder, "input_channels", 8), layers=args.decoder_layers).to(device)
    if args.init_decoder:
        if initial["metadata"]["encoder_sha256"] != encoder_hash or initial["words"] != words:
            raise ValueError("decoder initialization belongs to a different encoder or dictionary")
        expand_decoder(model, initial["model"])
        if "context_model" in initial:
            if not args.fine_tune_encoder:
                raise ValueError("a jointly trained decoder requires its learned context")
            encoder.load_state_dict(initial["context_model"])
            refresh_context(encoder, validation_paths, validation)
        del initial
    groups_to_train = [{"params": model.parameters(), "lr": args.learning_rate}]
    if args.fine_tune_encoder:
        groups_to_train.append({"params": encoder.parameters(), "lr": args.encoder_learning_rate})
    optimizer = torch.optim.AdamW(groups_to_train, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    metadata = {"encoder_sha256": encoder_hash, "encoder_path": str(args.checkpoint),
        "retrieval_sha256": retrieval_hash,
        "retrieval_role": "Frozen full-vocabulary negative mining and natural development candidates; context features remain separately bound",
        "decoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoder_initialization_sha256": decoder_hash,
        "fine_tuned_context": args.fine_tune_encoder,
        "context_normalization": "Fixed running statistics; encoder stays in evaluation mode during gradient updates",
        "objective": "Listwise candidate classification plus 0.25 times letter cross entropy" if args.ranking_candidates else "Full-spelling teacher-forced cross entropy",
        "arguments": {key: [str(p) for p in value] if key == "extra_train" else str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_corpora": [info for _, info in shards], "validation_corpus": val_info,
        "real_training_sha256": real_hash,
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
                    "epoch": -1, "best_score": best, "selected": chosen, "metadata": metadata, "words": words,
                    **({"context_model": encoder.state_dict()} if args.fine_tune_encoder else {})}, initial_path)
        initial_path.replace(args.output / "best.pt")
        (args.output / "best-metrics.json").write_text(json.dumps(initial_report, indent=2) + "\n")
    for epoch in range(args.epochs):
        model.train()
        began, accumulated = time.monotonic(), 0.0
        for step in range(args.steps):
            nr = args.batch * 15 // 100
            ns = args.batch // 4 if args.fresh_fraction is None else args.batch - nr - int(args.batch * args.fresh_fraction)
            nf = args.batch - nr - ns
            ri = balanced_real_indices(groups, nr)
            synthetic_paths, synthetic_labels = sample_corpora(shards, ns, rng, device)
            path_parts, label_parts = [real[ri], synthetic_paths], [real_labels[ri], synthetic_labels]
            if nf:
                fresh = sampler.labels(nf)
                path_parts.append(sampler.sample(fresh)); label_parts.append(fresh)
            paths, labels = torch.cat(path_parts), torch.cat(label_parts)
            if args.fine_tune_encoder:
                sequence, raw, embeddings = differentiable_context(encoder, paths)
            else:
                sequence, raw, embeddings = encode(encoder, paths)
            memory = model.encode_memory(sequence, raw, embeddings)
            logits = model(memory, inputs[labels])
            loss = F.cross_entropy(logits.flatten(0, 1), targets[labels].flatten(), ignore_index=0)
            if args.ranking_candidates:
                with torch.no_grad():
                    retrieval_query = retriever(encoder_features(retriever, paths)) if args.fine_tune_encoder or args.retrieval_checkpoint else embeddings
                    similarity = proxy_scores(retrieval_query, proxies)
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
            trainable = [*model.parameters()] + ([*encoder.parameters()] if args.fine_tune_encoder else [])
            torch.nn.utils.clip_grad_norm_(trainable, 5)
            optimizer.step()
            accumulated += loss.item()
        schedule.step()
        report = {"epoch": epoch + 1, "loss": accumulated / args.steps, "seconds": time.monotonic() - began}
        chosen, improved = None, False
        if epoch == 0 or (epoch + 1) % args.evaluate_every == 0:
            model.eval()
            if args.fine_tune_encoder:
                refresh_context(encoder, validation_paths, validation)
            chosen, trials = evaluate(model, validation, inputs, targets, lengths)
            score = chosen["validation"]["memory"]["top1"]
            improved = score > best
            best = max(best, score)
            report.update(selected=chosen, trials=trials, best_score=best, improved=improved)
        print(json.dumps({key: value for key, value in report.items() if key != "trials"}), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                      "epoch": epoch, "best_score": best, "selected": chosen, "metadata": metadata, "words": words,
                      **({"context_model": encoder.state_dict()} if args.fine_tune_encoder else {})}
        temporary = args.output / "latest.pt.tmp"
        torch.save(checkpoint, temporary); temporary.replace(args.output / "latest.pt")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save(checkpoint, temporary); temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
