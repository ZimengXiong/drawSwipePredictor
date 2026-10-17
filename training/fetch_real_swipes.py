#!/usr/bin/env python3
"""Fetch a bounded, reproducible sample of FUTO's MIT-licensed real swipe data."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "d71bf5fd7f45b3e7c2ed2d76a21b0dbd3b4ba566"
BASE = f"https://huggingface.co/datasets/futo-org/swipe.futo.org/resolve/{REVISION}"
SOURCE = "https://huggingface.co/datasets/futo-org/swipe.futo.org"
SIZES = {"train": 5157753763, "validation": 304264958, "test": 269612671}
FILES = {"train": "train.jsonl", "validation": "dev.jsonl", "test": "test.jsonl"}


def convert(row: dict, vocabulary: set[str], split: str) -> dict | None:
    """Validate recordings without filtering by recognition or template score."""
    word = row.get("word", "").lower().strip(" .,!?;:\"'()[]")
    if word not in vocabulary or not 4 <= len(word) <= 12:
        return None
    if not word.isascii() or not word.isalpha():
        return None
    width, height = float(row["canvas_width"]), float(row["canvas_height"])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        return None
    events = row.get("data", [])
    if len(events) < 6:
        return None
    points = [[float(p["x"]) * width, float(p["y"]) * height] for p in events]
    if not all(math.isfinite(v) for p in points for v in p):
        return None
    if max(p[0] for p in points) == min(p[0] for p in points) and max(p[1] for p in points) == min(p[1] for p in points):
        return None
    timestamps = [float(p["t"]) for p in events]
    if not all(math.isfinite(t) for t in timestamps) or any(a > b for a, b in zip(timestamps, timestamps[1:])):
        return None
    return {
        "word": word, "points": points, "times_ms": [t - timestamps[0] for t in timestamps],
        "source": "futo_swipe_1", "source_id": row["id"],
        "session": row["session"], "split": split,
    }


def fetch_window(split: str, start: int, size: int) -> bytes:
    end = min(start + size - 1, SIZES[split] - 1)
    # Unique query prevents intermediaries from reusing a different byte range.
    url = f"{BASE}/{FILES[split]}?window={start}"
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"server did not honor byte range: {response.status} {content_range}")
        data = response.read(size + 1)
    if len(data) > size:
        raise RuntimeError("response exceeded requested download size")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/data/futo")
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--vocab", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test, args.vocab) <= 0:
        parser.error("sample counts and vocabulary must be positive")
    vocabulary = []
    for line in (ROOT / "word_freq.txt").read_text().splitlines():
        word = line.split("\t")[0]
        if len(word) >= 2 and word.isascii() and word.isalpha() and len(set(word)) > 1:
            vocabulary.append(word)
        if len(vocabulary) == args.vocab:
            break
    vocabulary_set = set(vocabulary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".windows"
    cache.mkdir(exist_ok=True)
    metadata = []
    sessions = {}
    for index, split in enumerate(FILES):
        randomizer = random.Random(args.seed + index)
        count = getattr(args, split)
        selected, seen, windows = [], set(), []
        starts = [randomizer.randrange(0, SIZES[split] - 1024 * 1024) for _ in range(max(128, count // 50))]

        def cached_window(start: int) -> bytes:
            cached = cache / f"{split}-{start}.jsonl"
            if cached.exists():
                return cached.read_bytes()
            data = fetch_window(split, start, 1024 * 1024)
            cached.write_bytes(data)
            return data

        def window_stream():
            with ThreadPoolExecutor(max_workers=4) as pool:
                for batch in range(0, len(starts), 4):
                    yield from zip(starts[batch:batch + 4], pool.map(cached_window, starts[batch:batch + 4]))

        for start, data in window_stream():
            windows.append(start)
            # Discard partial records at both ends of the byte window.
            lines = data.split(b"\n")[1:-1]
            randomizer.shuffle(lines)
            for line in lines:
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                sample = convert(row, vocabulary_set, split)
                if sample:
                    selected.append(sample)
                if len(selected) == count:
                    break
            if len(windows) % 10 == 0:
                print(f"{split}: {len(selected)}/{count} recordings from {len(windows)} windows", flush=True)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError(f"only found {len(selected)} valid {split} records; lower the requested count")
        randomizer.shuffle(selected)
        path = output / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as target:
            for sample in selected:
                target.write(json.dumps(sample, separators=(",", ":")) + "\n")
        temporary.replace(path)
        sessions[split] = {row["session"] for row in selected}
        metadata.append({
            "split": split, "samples": count, "sessions": len(sessions[split]),
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_file": FILES[split], "byte_windows": windows,
        })
        print(f"{split}: {count} real gestures from {len(sessions[split])} donor sessions", flush=True)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not sessions[a].isdisjoint(sessions[b]):
            raise RuntimeError(f"source session split overlaps between {a} and {b}")
    for filename in ["LICENSE", "README.md"]:
        with urllib.request.urlopen(f"{BASE}/{filename}", timeout=60) as response:
            (output / f"FUTO-{filename}").write_bytes(response.read())
    manifest = {
        "schema_version": 1, "sampler_version": 2, "source": SOURCE, "revision": REVISION, "license": "MIT",
        "seed": args.seed, "vocabulary_limit": args.vocab, "splits": metadata,
        "sampling": "Random byte windows, then shuffled complete records; not a uniform sample of all rows.",
        "filters": "4–12 ASCII letters, in vocabulary, >=6 points, finite moving paths and monotone timestamps. No distance-score filter.",
        "coordinates": "Source normalized x/y multiplied by canvas_width/canvas_height.",
        "limitation": "Real on-keyboard swipes; not a dataset of freehand gestures drawn from memory.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
