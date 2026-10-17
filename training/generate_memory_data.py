#!/usr/bin/env python3
"""Generate labeled gestures with disjoint writer identities and target words."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/data/memory")
    parser.add_argument("--train", type=int, default=6000, help="samples per difficulty")
    parser.add_argument("--validation", type=int, default=1000, help="samples per difficulty")
    parser.add_argument("--test", type=int, default=1000, help="samples per difficulty")
    parser.add_argument("--vocab", type=int, default=10000)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1 or args.vocab < 100:
        parser.error("sample counts must be positive and vocabulary must be at least 100")
    if max(args.train, args.validation, args.test) >= 1_000_003:
        parser.error("use fewer than 1,000,003 samples per difficulty to keep writer splits disjoint")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "swipe-engine", "--example", "memory_bench"],
        cwd=ROOT, check=True,
    )
    train_words = args.vocab * 8 // 10
    validation_words = args.vocab // 10
    splits = [
        ("train", args.train, 101, 0, train_words),
        ("validation", args.validation, 102, train_words, validation_words),
        ("test", args.test, 9001, train_words + validation_words,
         args.vocab - train_words - validation_words),
    ]
    metadata = []
    for name, samples, seed, offset, words in splits:
        path = output / f"{name}.jsonl"
        report = output / f"{name}-generation.json"
        subprocess.run([
            str(ROOT / "target/release/examples/memory_bench"),
            "--export-only", "--dataset", str(path), "--output", str(report),
            "--samples", str(samples), "--seed", str(seed), "--vocab", str(args.vocab),
            "--target-offset", str(offset), "--target-vocab", str(words),
        ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        metadata.append({
            "split": name, "file": path.name, "samples": samples * 3,
            "seed": seed, "sha256": digest(path), "bytes": path.stat().st_size,
            "target_offset": offset, "target_vocabulary_limit": words,
        })
        print(f"{name}: {samples * 3:,} gestures saved to {path}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_version": 1, "generator_commit": commit, "synthetic_only": True,
        "vocabulary_limit": args.vocab, "splits": metadata,
        "split_policy": "Disjoint target-word ranges and writer seed ranges. Each writer draws the same word at three difficulty levels within its split.",
        "point_format": "Ordered [x, y] screen coordinates; labels are lowercase ASCII words.",
        "limitations": "Synthetic writers approximate memory errors. This dataset does not measure human recognition accuracy.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
