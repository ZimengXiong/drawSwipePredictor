#!/usr/bin/env python3
"""Build independent full-vocabulary writer shards with overlap and hash checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

from train_proxy import corpus

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 1_000_003


def overlaps(first, second):
    return not (first[1] <= second[0] or second[1] <= first[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-dir", type=Path, default=Path("data/memory-proxy"))
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.shards, args.samples, args.workers) < 1 or args.samples >= BLOCK or args.start_seed < 0:
        parser.error("use positive counts, fewer than one seed block per shard, and nonnegative seeds")
    seeds = list(range(args.start_seed, args.start_seed + args.shards))
    if seeds[-1] > (2**64 - 1) // BLOCK - 1:
        parser.error("writer seeds exceed the corpus format")
    protocol = json.loads((ROOT / "benchmarks/target-80-protocol.json").read_text())
    held = [(protocol["development"]["validation_seed"], protocol["development"]["validation_writers"]),
            (protocol["confirmation_holdout"]["seed"], protocol["confirmation_holdout"]["writers"])]
    occupied = [(seed * BLOCK, seed * BLOCK + count) for seed, count in held]
    targets = {str((args.output_dir / f"train-{seed}.bin").resolve()) for seed in seeds}
    for path in args.existing_dir.glob("train*.bin"):
        if str(path.resolve()) in targets:
            continue
        with path.open("rb") as stream:
            magic, count, size, seed = struct.unpack("<8sQIQ", stream.read(28))
        if magic != b"SWPCOR01" or size != 60000:
            raise ValueError("existing training corpus header differs")
        occupied.append((seed * BLOCK, seed * BLOCK + count))
    for seed in seeds:
        if seed in [101, 102, 9001, 12345, 246810, 700003, 700007]:
            raise ValueError("reserved evaluation seed")
        interval = (seed * BLOCK, seed * BLOCK + args.samples)
        if any(overlaps(interval, other) for other in occupied):
            raise ValueError(f"writer interval overlaps an existing or held-out pool: {seed}")
        occupied.append(interval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(not (args.output_dir / f"train-{seed}.bin").exists() for seed in seeds)
    if shutil.disk_usage(args.output_dir).free < missing * (28 + args.samples * 528) + 1024**3:
        raise RuntimeError("insufficient disk space for the requested shards and reserve")
    subprocess.run(["cargo", "build", "--release", "-p", "swipe-engine", "--example", "memory_corpus"], cwd=ROOT, check=True)
    cli = ROOT / "target/release/examples/memory_corpus"
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                     ["word_freq.txt", "crates/swipe-engine/src/synthetic.rs", "crates/swipe-engine/examples/memory_corpus.rs"]}
    cli_hash = hashlib.sha256(cli.read_bytes()).hexdigest()

    def build(seed):
        final = args.output_dir / f"train-{seed}.bin"
        receipt = final.with_suffix(".build.json")
        if final.exists():
            if not receipt.exists():
                raise ValueError(f"existing shard has no build receipt: {final}")
            record = json.loads(receipt.read_text())
            rows, info = corpus(final)
            del rows
            if record["source_sha256"] != source_hashes or record["corpus"] != info or info["seed"] != seed or info["samples"] != args.samples:
                raise ValueError("existing shard provenance or parameters differ")
            return record
        pending = final.with_suffix(".pending.bin")
        if pending.exists():
            raise RuntimeError(f"pending shard exists; inspect the original builder before restarting: {pending}")
        began = time.monotonic()
        subprocess.run([str(cli), "--output", str(pending), "--samples", str(args.samples), "--seed", str(seed), "--vocab", "60000"],
                       check=True, capture_output=True, text=True)
        rows, info = corpus(pending)
        del rows
        if info["seed"] != seed or info["samples"] != args.samples or info["vocabulary"] != 60000:
            raise ValueError("generated shard parameters differ")
        record = {"path": str(final), "corpus": info, "source_sha256": source_hashes,
                  "cli_sha256": cli_hash, "seconds": time.monotonic() - began}
        pending.replace(final)
        Path(str(pending) + ".json").replace(Path(str(final) + ".json"))
        temporary_receipt = receipt.with_suffix(".json.tmp")
        temporary_receipt.write_text(json.dumps(record, indent=2) + "\n")
        temporary_receipt.replace(receipt)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build, seed) for seed in seeds]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed_shards": len(records), "total_shards": args.shards,
                              "seed": record["corpus"]["seed"], "sha256": record["corpus"]["sha256"]}), flush=True)
    report = {"test_data_used": False, "full_vocabulary": 60000, "generated_writers": args.samples * args.shards,
              "shards": sorted(records, key=lambda row: row["corpus"]["seed"]),
              "held_writer_intervals": occupied[:len(held)], "source_sha256": source_hashes,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "rules": "Uniform eligible target words across the full dictionary; unchanged writer errors; independent seed blocks; no development or benchmark drawings used"}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
