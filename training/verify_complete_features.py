#!/usr/bin/env python3
"""Check raw-writer replay and label-independent complete geometry features."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from retrieval_model import vocabulary
from train_proxy import ROOT, RECORD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=ROOT / "target/release/examples/retrieval_cli")
    parser.add_argument("--replay-cli", type=Path, default=ROOT / "target/release/examples/replay_memory_writers")
    parser.add_argument("--corpus-cli", type=Path, default=ROOT / "target/release/examples/memory_corpus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.model / "manifest.json").read_text())
    train = ROOT / "training/data/memory-proxy/train.bin"
    with train.open("rb") as source:
        data_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if data_hash != manifest["training"]["train_corpus"]["sha256"]:
        raise ValueError("raw replay must use the declared original training corpus")
    rows = np.memmap(train, dtype=RECORD, mode="r", offset=28, shape=(12,))
    words, _, _ = vocabulary(ROOT / "word_freq.txt", 60000)
    requests = [{"word": words[int(row["label"])], "writer": int(row["writer"]),
                 "difficulty": ["clean", "loose", "memory"][int(row["difficulty"])]} for row in rows]
    generated = subprocess.run([str(args.replay_cli.resolve())], input="".join(json.dumps(row) + "\n" for row in requests),
                               capture_output=True, text=True, check=True)
    replayed = [json.loads(line) for line in generated.stdout.splitlines()]
    if len(replayed) != len(rows) or any(set(row) != {"points"} for row in replayed):
        raise ValueError("raw replay must return every stroke without metadata")
    with tempfile.TemporaryDirectory(prefix="drawswipe-raw-replay-") as directory:
        binary, reference = Path(directory) / "corpus.bin", Path(directory) / "raw.json"
        subprocess.run([str(args.corpus_cli.resolve()), "--output", str(binary), "--samples", "12", "--seed", "700001",
                        "--vocab", "60000", "--verification", str(reference), "--verification-limit", "12"],
                       capture_output=True, text=True, check=True)
        if binary.read_bytes()[28:] != rows.tobytes():
            raise ValueError("original training records do not regenerate exactly")
        expected = json.loads(reference.read_text())
        if any(actual["points"] != original["points"] for actual, original in zip(replayed, expected)):
            raise ValueError("raw replay differs from the unchanged generator")
    point_request = replayed[3]
    adversarial = [{**point_request, "word": word, "writer": writer} for word, writer in [("ignore_me", 0), ("another_label", 999)]]
    features = subprocess.run([str(args.cli.resolve()), "--model", str(args.model.resolve()), "--complete-features", "--candidates", "8"],
                              input="".join(json.dumps(row) + "\n" for row in adversarial), capture_output=True, text=True, check=True)
    responses = [json.loads(line) for line in features.stdout.splitlines()]
    if len(responses) != 2 or responses[0] != responses[1] or responses[0]["schema"] != "complete_geometry_v1":
        raise ValueError("candidate features depend on ignored metadata")
    candidates = responses[0]["candidates"]
    if len(candidates) != 8 or any(len(row["features"]) != 20 or row["features"][2] != 0 for row in candidates):
        raise ValueError("unexpected feature schema or popularity signal")
    if not np.isfinite([row["features"] for row in candidates]).all():
        raise ValueError("nonfinite candidate features")
    normal = subprocess.run([str(args.cli.resolve()), "--model", str(args.model.resolve())], input=json.dumps(point_request) + "\n",
                            capture_output=True, text=True, check=True)
    predictions = json.loads(normal.stdout)["predictions"]
    if [row["word"] for row in candidates[:5]] != [row["word"] for row in predictions]:
        raise ValueError("feature extraction changed the complete candidate order")
    if any(np.float32(row["features"][1]) != np.float32(prediction["rating"]) for row, prediction in zip(candidates, predictions)):
        raise ValueError("feature extraction changed the complete scores")
    report = {"test_data_used": False, "training_sha256": data_hash, "raw_replay_cases": len(rows),
              "native_cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
              "replay_cli_sha256": hashlib.sha256(args.replay_cli.read_bytes()).hexdigest(),
              "source_manifest_sha256": hashlib.sha256((args.model / "manifest.json").read_bytes()).hexdigest(),
              "feature_schema": "complete_geometry_v1", "features": 20, "popularity_feature_is_zero": True,
              "metadata_independence_verified": True, "complete_scores_and_top5_unchanged": True,
              "limits": "Training-data replay and feature implementation checks only. No new ranker, development accuracy, benchmark evaluation, or confirmation evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
