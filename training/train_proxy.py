#!/usr/bin/env python3
"""Learn full-vocabulary word proxies from fresh synthetic writers and real traces."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from motor_sampler import MotorSampler
from retrieval_model import GestureEncoder, augment, build_encoder, encoder_features, features, load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]
RECORD = np.dtype([("label", "<u4"), ("difficulty", "<u4"), ("writer", "<u8"), ("points", "<f4", (64, 2))])


def corpus(path):
    with path.open("rb") as source:
        magic, count, size, seed = struct.unpack("<8sQIQ", source.read(28))
    if magic != b"SWPCOR01" or path.stat().st_size != 28 + count * RECORD.itemsize:
        raise ValueError("invalid synthetic corpus format")
    rows = np.memmap(path, dtype=RECORD, mode="r", offset=28, shape=(count,))
    if rows["label"].max() >= size or rows["difficulty"].max() > 2 or not np.isfinite(rows["points"]).all():
        raise ValueError("invalid synthetic corpus values")
    if not np.array_equal(rows["writer"], seed * 1_000_003 + np.arange(count, dtype=np.uint64)):
        raise ValueError("unexpected writer identities")
    return rows, {"samples": count, "vocabulary": size, "seed": seed,
                  "writer_start": seed * 1_000_003, "writer_end": seed * 1_000_003 + count,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sample_corpora(shards, count, rng, device):
    """Read a uniform batch from memory-mapped training shards."""
    edges = np.cumsum([len(rows) for rows, _ in shards])
    selected = rng.integers(int(edges[-1]), size=count)
    shard_ids = np.searchsorted(edges, selected, side="right")
    points = np.empty((count, 64, 2), dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    starts = np.r_[0, edges[:-1]]
    for i, (rows, _) in enumerate(shards):
        mask = shard_ids == i
        local = selected[mask] - starts[i]
        points[mask], labels[mask] = rows["points"][local], rows["label"][local]
    return torch.tensor(points, device=device), torch.tensor(labels, device=device)


def real_word_groups(labels, device):
    """Keep every recording, grouped by its training word label."""
    order = np.argsort(labels, kind="stable")
    _, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    return tuple(torch.tensor(values, dtype=torch.long, device=device) for values in [order, starts, counts])


def balanced_real_indices(groups, count):
    """Sample observed word types uniformly, then a recording of that word."""
    order, starts, counts = groups
    chosen = torch.randint(len(starts), (count,), device=order.device)
    within = (torch.rand(count, device=order.device) * counts[chosen]).long()
    return order[starts[chosen] + within]


def proxy_scores(query, proxies):
    """Score one or several equally weighted route centers per word."""
    normalized = F.normalize(proxies, dim=-1)
    if normalized.ndim == 2:
        return query @ normalized.T
    similarities = (query @ normalized.reshape(-1, normalized.shape[-1]).T).reshape(len(query), len(proxies), proxies.shape[1])
    weights = F.softmax(similarities / 0.1, dim=2)
    return (weights * similarities).sum(2)


def proxy_checkpoint_state(proxies):
    """Keep inference vectors and the raw parameters needed by the optimizer."""
    raw = proxies.detach().clone()
    return {"proxies": F.normalize(raw, dim=-1), "raw_proxies": raw}


def training_proxy_vectors(saved):
    """Load raw state when available; retain compatibility with older files."""
    normalized = saved["proxies"]
    raw = saved.get("raw_proxies")
    if raw is None:
        return normalized
    if raw.shape != normalized.shape or raw.dtype != normalized.dtype or not torch.isfinite(raw).all():
        raise ValueError("raw word vectors differ from inference vectors")
    if not torch.allclose(F.normalize(raw, dim=-1), normalized, atol=1e-6, rtol=1e-5):
        raise ValueError("raw word vectors differ from inference vectors")
    return raw


@torch.no_grad()
def evaluate(model, proxies, datasets):
    result = {}
    for name, (paths, labels) in datasets.items():
        hits = []
        for start in range(0, len(paths), 256):
            ranked = proxy_scores(model(encoder_features(model, paths[start:start + 256])), proxies).topk(5, dim=1).indices
            truth = labels[start:start + 256, None]
            hits.append(torch.stack([(ranked[:, :1] == truth).any(1), (ranked == truth).any(1)], 1))
        top1, top5 = torch.cat(hits).float().mean(0).tolist()
        result[name] = {"samples": len(paths), "top1": top1, "top5": top5}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "training/data/memory-proxy/train.bin")
    parser.add_argument("--extra-train", type=Path, action="append", default=[])
    parser.add_argument("--stream-corpus", action="store_true")
    parser.add_argument("--balance-real", action="store_true")
    parser.add_argument("--freeze-proxies", action="store_true")
    parser.add_argument("--validation", type=Path, default=ROOT / "training/data/memory-proxy/validation.bin")
    parser.add_argument("--init-model", type=Path, default=ROOT / "training/checkpoints/retrieval-v3/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--proxy-learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=812831)
    parser.add_argument("--architecture", choices=["small", "large", "intrinsic", "giant", "attention", "wide512"], default="small")
    parser.add_argument("--centers", type=int, default=1)
    parser.add_argument("--proxy-init", type=Path)
    parser.add_argument("--fresh-fraction", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch) < 1:
        parser.error("counts must be positive")
    rates = [args.learning_rate] + ([] if args.proxy_learning_rate is None else [args.proxy_learning_rate])
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("learning rates must be finite and positive")
    if not 1 <= args.centers <= 8:
        parser.error("use 1 to 8 route centers per word")
    if not 0 <= args.fresh_fraction <= 0.7:
        parser.error("fresh fraction must be between 0 and 0.7")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.75)
    train, train_info = corpus(args.train)
    shards = [(train, train_info)] + [corpus(path) for path in args.extra_train]
    rng = np.random.default_rng(args.seed)
    validation, validation_info = corpus(args.validation)
    declared = [info for _, info in shards] + [validation_info]
    for i, record in enumerate(declared):
        if record["seed"] in [101, 102, 9001, 12345, 246810, 700007]:
            raise ValueError("reserved benchmark writer block")
        if record["vocabulary"] != train_info["vocabulary"]:
            raise ValueError("corpus vocabularies differ")
        for other in declared[:i]:
            if not (record["writer_end"] <= other["writer_start"] or other["writer_end"] <= record["writer_start"]):
                raise ValueError("declared writer ranges overlap")
    if args.extra_train and not args.stream_corpus:
        parser.error("additional training shards require --stream-corpus")
    words, counts, canonical = vocabulary(ROOT / "word_freq.txt", train_info["vocabulary"])
    if len(words) != validation_info["vocabulary"]:
        raise ValueError("corpus vocabularies differ")
    ids = {word: i for i, word in enumerate(words)}
    real_path = ROOT / "training/data/futo-large/train.jsonl"
    real, real_labels = load_recordings(real_path, ids)
    real_groups = real_word_groups(real_labels, device) if args.balance_real else None
    real = torch.tensor(real, device=device)
    real_labels = torch.tensor(real_labels, device=device)
    if not args.stream_corpus:
        train_paths = torch.tensor(np.array(train["points"]), device=device)
        train_labels = torch.tensor(np.array(train["label"], dtype=np.int64), device=device)
    canonical = torch.tensor(canonical, device=device)
    datasets = {}
    for difficulty, name in enumerate(["clean", "loose", "memory"]):
        mask = validation["difficulty"] == difficulty
        datasets[name] = (torch.tensor(np.array(validation["points"][mask]), device=device), torch.tensor(np.array(validation["label"][mask], dtype=np.int64), device=device))
    model = build_encoder(args.architecture).to(device)
    initial_content = args.init_model.read_bytes()
    initial_hash = hashlib.sha256(initial_content).hexdigest()
    initial = torch.load(io.BytesIO(initial_content), map_location="cpu", weights_only=False)
    del initial_content
    if initial["words"] != words:
        raise ValueError("initial model vocabulary differs from the training corpus")
    initial_architecture = initial["metadata"]["arguments"].get("architecture", "small")
    teacher = None
    if args.architecture != initial_architecture:
        if args.architecture == "wide512" and (initial_architecture != "large" or args.freeze_proxies):
            raise ValueError("expanded word space requires a large initializer and trainable word vectors")
        if args.architecture == "attention" and initial_architecture != "large":
            raise ValueError("attention corrections require a completed large initializer")
        if args.architecture == "attention":
            model.base.load_state_dict(initial["model"])
        else:
            compatible = {key: value for key, value in initial["model"].items() if key in model.state_dict() and model.state_dict()[key].shape == value.shape}
            model.load_state_dict(compatible, strict=False)
        if args.architecture == "giant":
            from large_encoder import expand_encoder
            expand_encoder(model, initial["model"])
        if args.architecture == "wide512":
            from large_encoder import expand_word_space
            expand_word_space(model, initial["model"])
        if args.architecture == "intrinsic" and initial["model"]["stages.0.weight"].shape[1] == 8:
            with torch.no_grad():
                model.stages[0].weight[:, :8].copy_(initial["model"]["stages.0.weight"])
                model.stages[0].weight[:, 8:].zero_()
        teacher = build_encoder(initial_architecture).to(device).eval()
        teacher.load_state_dict(initial["model"])
        teacher.requires_grad_(False)
    else:
        model.load_state_dict(initial["model"])
    model.eval()
    sampler = MotorSampler(words, device) if args.fresh_fraction else None
    with torch.no_grad():
        initial_vectors = training_proxy_vectors(initial).to(device) if "proxies" in initial else torch.cat([model(encoder_features(model, part)) for part in canonical.split(512)])
    preserve_raw = "raw_proxies" in initial
    if args.proxy_init:
        loaded = np.load(args.proxy_init, allow_pickle=False)
        if loaded["words"].tolist() != words:
            raise ValueError("proxy initialization vocabulary differs")
        initial_vectors = torch.tensor(loaded["vectors"], device=device)
        preserve_raw = False
    elif args.centers > 1 and initial_vectors.ndim == 2:
        initial_vectors = initial_vectors[:, None].repeat(1, args.centers, 1)
        initial_vectors = initial_vectors + torch.randn_like(initial_vectors) * 0.01
        preserve_raw = False
    embedding_dim = getattr(model, "embedding_dim", 128)
    if args.architecture == "wide512" and initial_vectors.shape[-1] == 128:
        initial_vectors = F.pad(initial_vectors, (0, embedding_dim - 128))
    expected = (len(words), embedding_dim) if args.centers == 1 else (len(words), args.centers, embedding_dim)
    if tuple(initial_vectors.shape) != expected:
        raise ValueError("proxy center dimensions differ from the requested model")
    proxies = nn.Parameter(initial_vectors if preserve_raw else F.normalize(initial_vectors, dim=-1), requires_grad=not args.freeze_proxies)
    del initial, initial_vectors
    proxy_rate = args.proxy_learning_rate if args.proxy_learning_rate is not None else args.learning_rate * 4
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": args.learning_rate},
                                  {"params": [proxies], "lr": proxy_rate}], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    start, best = 0, -1.0
    legacy_resume = False
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["words"] != words or saved["metadata"]["validation_corpus"]["sha256"] != validation_info["sha256"]:
            raise ValueError("resume checkpoint vocabulary or development writers differ")
        recorded = [saved["metadata"]["train_corpus"]] + saved["metadata"].get("extra_train_corpora", [])
        if [record["sha256"] for record in recorded] != [info["sha256"] for _, info in shards]:
            raise ValueError("resume checkpoint training shards differ")
        model.load_state_dict(saved["model"])
        proxies.data.copy_(training_proxy_vectors(saved))
        legacy_resume = "raw_proxies" not in saved
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"] + 1, saved["best_score"]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": {k: [str(item) for item in v] if k == "extra_train" else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_corpus": train_info, "extra_train_corpora": [info for _, info in shards[1:]], "validation_corpus": validation_info,
        "real_training_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "real_sampling": "uniform observed word types; all recordings retained" if args.balance_real else "uniform recordings",
        "initialization_sha256": initial_hash,
        "proxy_state_schema": 2, "initialization_preserves_raw_vectors": preserve_raw,
        "legacy_normalized_resume": legacy_resume,
        "optimizer_learning_rates_at_start": [group["lr"] for group in optimizer.param_groups],
        "test_data_used": False, "selection": "uniform full-vocabulary memory validation top1",
        "input_features": "geometry only; labels and writer identities are never model inputs",
        "motor_sampler_sha256": hashlib.sha256((ROOT / "training/motor_sampler.py").read_bytes()).hexdigest() if sampler else None,
        "proxy_initialization_sha256": hashlib.sha256(args.proxy_init.read_bytes()).hexdigest() if args.proxy_init else None,
        "resume_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    initial_metrics = evaluate(model, proxies, datasets)
    print(json.dumps({"initial_validation": initial_metrics}), flush=True)
    if start == 0:
        best = initial_metrics["memory"]["top1"]
        baseline = {"model": model.state_dict(), **proxy_checkpoint_state(proxies),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": -1,
            "best_score": best, "words": words, "counts": counts, "metadata": metadata}
        temporary = args.output / "best.pt.tmp"
        torch.save(baseline, temporary); temporary.replace(args.output / "best.pt")
        (args.output / "best-metrics.json").write_text(json.dumps({"epoch": 0, "validation": initial_metrics, "best_score": best}, indent=2) + "\n")
        del baseline
    for epoch in range(start, args.epochs):
        model.train()
        began, loss_sum = time.monotonic(), 0.0
        for step in range(args.steps):
            nr, nc = args.batch * 15 // 100, args.batch // 10
            nf = int(args.batch * args.fresh_fraction)
            ns = args.batch - nr - nc - nf
            ri = balanced_real_indices(real_groups, nr) if args.balance_real else torch.randint(len(real), (nr,), device=device)
            if args.stream_corpus:
                synthetic_paths, synthetic_labels = sample_corpora(shards, ns, rng, device)
            else:
                si = torch.randint(len(train_paths), (ns,), device=device)
                synthetic_paths, synthetic_labels = train_paths[si], train_labels[si]
            ci = torch.randint(len(words), (nc,), device=device)
            # Procedural drawings already contain strong memory errors. Avoid
            # destroying their remaining evidence with a second heavy distortion.
            paths = torch.cat([augment(real[ri]), synthetic_paths, augment(canonical[ci])])
            labels = torch.cat([real_labels[ri], synthetic_labels, ci])
            if sampler:
                fresh_labels = sampler.labels(nf)
                paths = torch.cat([paths, sampler.sample(fresh_labels)])
                labels = torch.cat([labels, fresh_labels])
            query = model(encoder_features(model, paths))
            logits = 32.0 * proxy_scores(query, proxies)
            loss = F.cross_entropy(logits, labels)
            if teacher is not None:
                with torch.no_grad():
                    target_vectors = teacher(encoder_features(teacher, paths))
                # Preserve the useful coarse geometry while the larger model
                # learns evidence that the smaller encoder cannot represent.
                weight = 2.0 if epoch < 5 else 0.1
                old_view = F.normalize(query[:, :target_vectors.shape[1]], dim=1) if query.shape[1] != target_vectors.shape[1] else query
                loss = loss + weight * (1 - (old_view * target_vectors).sum(1)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite proxy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), proxies], 5)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        model.eval()
        metrics = evaluate(model, proxies, datasets)
        score = metrics["memory"]["top1"]
        improved = score > best
        best = max(best, score)
        result = {"epoch": epoch + 1, "loss": loss_sum / args.steps, "validation": metrics,
                  "best_score": best, "seconds": time.monotonic() - began, "improved": improved}
        print(json.dumps(result), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(result) + "\n")
        saved = {"model": model.state_dict(), **proxy_checkpoint_state(proxies),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best_score": best, "words": words, "counts": counts, "metadata": metadata}
        temporary = args.output / "latest.pt.tmp"
        torch.save(saved, temporary)
        temporary.replace(args.output / "latest.pt")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save(saved, temporary)
            temporary.replace(args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
