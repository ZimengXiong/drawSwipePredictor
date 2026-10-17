#!/usr/bin/env python3
"""Add pinned public human recordings while excluding held-out sessions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from fetch_real_swipes import BASE, REVISION, SOURCE, ROOT, convert

FILES = {
    2: (105042658, "c59beca97ba7014b3a882597208af115de0d38390960d1f91bb74153574dc967"),
    3: (199473677, "6e9f6dde97553077352faba30b309e3d7f2f129c53ee82e4d28fe72c8743f95a"),
    4: (436344849, "255994e4c79c42ec28656e2cbc5573b39a6c07087bc80a5361ff82339f109ca5"),
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def training_record(row, vocabulary, collection, blocked_sessions):
    """Exclude held sessions before applying score-independent validity checks."""
    if row["session"] in blocked_sessions:
        return None
    sample = convert(row, vocabulary, "train")
    if sample is not None:
        sample["source"] = f"futo_swipe_{collection}"
        sample["source_id"] = f"{collection}:{row['id']}"
    return sample


def download(collection, directory):
    filename = f"swipe-{collection}/swipe{collection}.jsonl"
    destination = directory / f"swipe{collection}.jsonl"
    size, expected = FILES[collection]
    if destination.exists():
        if destination.stat().st_size != size or digest(destination) != expected:
            raise ValueError("cached source differs from the pinned dataset")
        return collection, destination
    pending = destination.with_suffix(".partial")
    if pending.exists():
        raise ValueError("partial download exists; inspect its original process before retrying")
    with urllib.request.urlopen(f"{BASE}/{filename}?download=true", timeout=60) as response, pending.open("wb") as target:
        shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
    if pending.stat().st_size != size or digest(pending) != expected:
        raise ValueError("download differs from the pinned source size or SHA256")
    pending.replace(destination)
    print(json.dumps({"downloaded_collection": collection, "bytes": size, "sha256": expected}), flush=True)
    return collection, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "training/data/futo-large")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/data/futo-expanded-v1")
    args = parser.parse_args()
    base_manifest = json.loads((args.base_dir / "manifest.json").read_text())
    if base_manifest["revision"] != REVISION:
        raise ValueError("base and additional sources must use the same pinned revision")
    split_records = {record["split"]: record for record in base_manifest["splits"]}
    blocked, held_hashes = set(), {}
    for name in ["validation", "test"]:
        path = args.base_dir / split_records[name]["file"]
        actual = digest(path)
        if actual != split_records[name]["sha256"]:
            raise ValueError("held-out session source hash differs")
        held_hashes[name] = actual
        with path.open() as source:
            for line in source:
                blocked.add(json.loads(line)["session"])
    original = args.base_dir / split_records["train"]["file"]
    if digest(original) != split_records["train"]["sha256"]:
        raise ValueError("original training data hash differs")
    vocabulary = []
    for line in (ROOT / "word_freq.txt").read_text().splitlines():
        word = line.split("\t")[0]
        if len(word) >= 2 and word.isascii() and word.islower() and word.isalpha() and len(set(word)) > 1:
            vocabulary.append(word)
        if len(vocabulary) == 60000:
            break
    if len(vocabulary) != 60000:
        raise ValueError("full deployment vocabulary is unavailable")
    words = set(vocabulary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "sources"
    cache.mkdir()
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources = list(pool.map(lambda collection: download(collection, cache), FILES))
    counts, sessions, baseline_sessions = Counter(), set(), set()
    with original.open() as source:
        for line in source:
            row = json.loads(line)
            baseline_sessions.add(row["session"])
            counts[row["word"]] += 1
    if baseline_sessions & blocked:
        raise ValueError("base training and held-out sessions overlap")
    initial_types, records = len(counts), []
    pending = output / "train.jsonl.tmp"
    with pending.open("wb") as target, original.open("rb") as source:
        shutil.copyfileobj(source, target)
        for collection, path in sources:
            seen, rejected, accepted, new_types = set(), Counter(), 0, set()
            with path.open() as incoming:
                for line in incoming:
                    try:
                        row = json.loads(line)
                        if row["id"] in seen:
                            rejected["duplicate_source_id"] += 1
                            continue
                        seen.add(row["id"])
                        if row["session"] in blocked:
                            rejected["held_out_session"] += 1
                            continue
                        sample = training_record(row, words, collection, blocked)
                        if sample is None:
                            rejected["outside_vocabulary_or_invalid_geometry"] += 1
                            continue
                    except (KeyError, ValueError, TypeError, OverflowError):
                        rejected["malformed_record"] += 1
                        continue
                    target.write((json.dumps(sample, separators=(",", ":")) + "\n").encode())
                    accepted += 1
                    new_types.add(sample["word"])
                    counts[sample["word"]] += 1
                    sessions.add(sample["session"])
            records.append({"collection": collection, "source_file": f"swipe-{collection}/swipe{collection}.jsonl",
                            "source_sha256": FILES[collection][1], "accepted": accepted,
                            "word_types": len(new_types), "rejected": dict(rejected)})
            print(json.dumps(records[-1]), flush=True)
    destination = output / "train.jsonl"
    pending.replace(destination)
    if sessions & blocked:
        raise ValueError("additional training and held-out sessions overlap")
    for filename in ["FUTO-LICENSE", "FUTO-README.md"]:
        shutil.copy2(args.base_dir / filename, output / filename)
    manifest = {"source": SOURCE, "revision": REVISION, "license": "MIT", "test_data_used_for_training": False,
                "original_train_sha256": split_records["train"]["sha256"], "held_out_file_sha256": held_hashes,
                "held_out_session_count": len(blocked), "held_out_session_overlap": 0,
                "additional_sessions": len(sessions), "all_training_sessions": len(sessions | baseline_sessions),
                "baseline_rows": split_records["train"]["samples"], "total_training_rows": sum(counts.values()),
                "baseline_word_types": initial_types, "total_word_types": len(counts), "full_vocabulary": len(words),
                "collections": records, "train_sha256": digest(destination),
                "importer_sha256": digest(Path(__file__)), "geometry_converter_sha256": digest(Path(__file__).with_name("fetch_real_swipes.py")),
                "policy": "Original training rows plus every valid full-vocabulary 4–12-letter recording from collections2–4, after held-session exclusion. No distance-score or model-performance filter. Held files supply session identifiers only; their words and trajectories never enter training.",
                "limits": "These are visible-keyboard human recordings, not freehand-memory labels. Additional collections contain unverified user errors. Session separation does not prove distinct human identities across sessions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["total_training_rows", "total_word_types", "held_out_session_overlap", "train_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
