#!/usr/bin/env python3
"""Learn a motor-fit initializer from fresh synthetic training geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from amortized_motor import AmortizedMotor, retained_training_keys
from generative_fit import MotorFitter
from motor_sampler import MotorSampler
from retrieval_model import vocabulary
from train_proxy import corpus

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def conditional_diagnostic(model, fitter, paths, labels):
    """Measure reconstruction given the correct word, not recognition accuracy."""
    errors = []
    for start in range(0, len(paths), 8):
        part = paths[start:start + 8]
        keys, lengths, valid, variants = fitter.hypotheses(labels[start:start + 8, None])
        keep = valid.reshape(-1)
        expanded = part[:, None].expand(-1, variants, -1, -1).reshape(-1, 64, 2)[keep]
        keys, lengths = keys[keep], lengths[keep]
        parameters = model(expanded, keys, lengths, fitter)
        error = (fitter.render(keys, lengths, parameters) - expanded).square().mean((1, 2))
        padded = torch.full((keep.numel(),), torch.inf, device=paths.device)
        padded[keep] = error
        errors.append(padded.reshape(len(part), variants).min(1).values)
    return float(torch.cat(errors).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=900379)
    parser.add_argument("--evaluate-every", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch, args.evaluate_every) < 1:
        parser.error("training counts must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    words, _, _ = vocabulary(ROOT / "word_freq.txt", 60000)
    fitter, sampler = MotorFitter(words), MotorSampler(words, "mps")
    width = fitter.keys.shape[1]
    model = AmortizedMotor(width).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    rows, validation = corpus(ROOT / "training/data/memory-proxy/validation.bin")
    protocol = json.loads((ROOT / "benchmarks/target-80-protocol.json").read_text())
    audit = json.loads((ROOT / "benchmarks/target-80-data-audit.json").read_text())
    if validation["seed"] != protocol["development"]["validation_seed"] or validation["sha256"] != audit["corpora"]["validation"]["sha256"] or validation["vocabulary"] != len(words):
        raise ValueError("motor diagnostics require the declared development writers")
    selected = np.flatnonzero(rows["difficulty"] == 2)[:256]
    val_paths = torch.tensor(np.array(rows["points"][selected]), device="mps")
    val_labels = torch.tensor(np.array(rows["label"][selected], dtype=np.int64), device="mps")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "kind": "amortized_motor_initializer", "recognition_accuracy": None, "test_data_used": False,
        "training": "Fresh independent memory-strength motor draws across the full eligible vocabulary; known retained routes supervise training reconstruction only",
        "validation": validation, "diagnostic": "Conditional reconstruction given the true word; not word-recognition accuracy",
        "source_sha256": {name: hashlib.sha256((ROOT / "training" / name).read_bytes()).hexdigest()
            for name in ["amortized_motor.py", "generative_fit.py", "motor_sampler.py"]},
        "word_frequency_sha256": hashlib.sha256((ROOT / "word_freq.txt").read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.eval()
    print(json.dumps({"initial_conditional_mse": conditional_diagnostic(model, fitter, val_paths, val_labels)}), flush=True)
    best = float("inf")
    for epoch in range(args.epochs):
        began, accumulated = time.monotonic(), 0.0
        model.train()
        for step in range(args.steps):
            labels = sampler.labels(args.batch)
            paths, known = sampler.sample(labels, strength=1.0, return_metadata=True)
            keys, lengths = retained_training_keys(known, width)
            parameters = model(paths, keys, lengths, fitter)
            reconstructed = fitter.render(keys, lengths, parameters)
            error = (reconstructed - paths).square().mean()
            penalty = parameters.tanh().square().mean()
            loss = 1000 * (error + 0.0002 * penalty)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite motor reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            accumulated += error.item()
        schedule.step()
        report = {"epoch": epoch + 1, "training_mse": accumulated / args.steps, "seconds": time.monotonic() - began}
        improved = False
        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0:
            model.eval()
            score = conditional_diagnostic(model, fitter, val_paths, val_labels)
            report["conditional_development_mse"] = score
            improved = score < best
            best = min(best, score)
        print(json.dumps(report), flush=True)
        with (args.output / "history.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "max_keys": width, "metadata": metadata, "best_conditional_mse": best, "words": words}
        temporary = args.output / "latest.pt.tmp"
        torch.save(saved, temporary); temporary.replace(args.output / "latest.pt")
        if improved:
            temporary = args.output / "best.pt.tmp"
            torch.save(saved, temporary); temporary.replace(args.output / "best.pt")


if __name__ == "__main__":
    main()
