#!/usr/bin/env python3
"""Mine candidate mistakes with the deployed encoder and native geometry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from retrieval_model import augment, load_recordings, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--samples", type=int, default=12000, help="samples per source")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "training/data/memory-wide")
    parser.add_argument("--real-dir", type=Path, default=ROOT / "training/data/futo-large")
    parser.add_argument("--model", type=Path, default=ROOT / "crates/swipe-engine/models/retrieval")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1 or not 1 <= args.workers <= 16:
        parser.error("use positive sample counts and 1 to 16 workers")
    torch.set_num_threads(4)
    seed = args.seed + (args.split == "validation")
    torch.manual_seed(seed)
    chooser = np.random.default_rng(seed)
    words, _, canonical = vocabulary(ROOT / "word_freq.txt", 60000)
    ids = {word: i for i, word in enumerate(words)}
    groups, paths, targets, hashes = [], [], [], {}
    for name, directory in [("real", args.real_dir), ("synthetic", args.synthetic_dir)]:
        source = directory / f"{args.split}.jsonl"
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        curves, labels = load_recordings(source, ids)
        selected = chooser.choice(len(curves), min(args.samples, len(curves)), replace=False)
        curves, labels = curves[selected], labels[selected]
        paths.append(curves)
        targets.extend(labels)
        groups.extend([name] * len(curves))
        if name == "real":
            # Separate noisy views teach the ranker when local alignment helps.
            distorted = augment(torch.tensor(curves, device="mps"), force_memory=True).cpu().numpy()
            paths.append(distorted)
            targets.extend(labels)
            groups.extend(["distorted_real"] * len(curves))
    labels = chooser.choice(len(words), args.samples, replace=False)
    paths.append(augment(torch.tensor(canonical[labels], device="mps"), force_memory=True).cpu().numpy())
    targets.extend(labels)
    groups.extend(["online"] * len(labels))
    paths = np.concatenate(paths)
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    binary = ROOT / "target/release/examples/retrieval_cli"
    features = np.zeros((len(paths), args.candidates, 20), dtype=np.float32)
    candidates = np.full((len(paths), args.candidates), -1, dtype=np.int32)

    def extract(indices):
        with tempfile.TemporaryFile(mode="w+") as inputs:
            for i in indices:
                inputs.write(json.dumps({"word": words[targets[i]], "points": paths[i].tolist()}) + "\n")
            inputs.seek(0)
            process = subprocess.Popen([str(binary), "--model", str(args.model), "--features", "--candidates", str(args.candidates)], stdin=inputs, stdout=subprocess.PIPE, text=True)
            received = 0
            for i, line in zip(indices, process.stdout):
                result = json.loads(line)["candidates"]
                if len(result) != args.candidates:
                    raise RuntimeError("native retrieval returned an incomplete shortlist")
                features[i] = [c["features"] for c in result]
                candidates[i] = [ids[c["word"]] for c in result]
                received += 1
                if received % 2000 == 0:
                    print(json.dumps({"worker": int(indices[0]), "processed": received}), flush=True)
            if process.wait() or received != len(indices):
                raise RuntimeError("native feature extraction failed")

    with ThreadPoolExecutor(args.workers) as executor:
        list(executor.map(extract, np.array_split(np.arange(len(paths)), args.workers)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, candidates=candidates, labels=targets, groups=groups)
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "split": args.split, "seed": seed, "source_sha256": hashes,
              "encoder_sha256": hashlib.sha256((args.model / "encoder.bin").read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256((args.model / "words.bin").read_bytes()).hexdigest(),
              "test_data_used": False, "candidates": args.candidates, "candidate_policy": "half neural retrieval, half independent geometry retrieval", "samples": len(paths), "candidate_recall": {}}
    for name in np.unique(groups):
        mask = groups == name
        report["candidate_recall"][name] = float((candidates[mask] == targets[mask, None]).any(axis=1).mean())
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
