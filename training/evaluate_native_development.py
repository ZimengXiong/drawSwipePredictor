#!/usr/bin/env python3
"""Replay every declared development stroke through the native CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from train_proxy import corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--corpus-cli", type=Path, default=Path("../target/release/examples/memory_corpus"))
    parser.add_argument("--retrieval-cli", type=Path, default=Path("../target/release/examples/retrieval_cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.model / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    cli_hash = hashlib.sha256(args.retrieval_cli.read_bytes()).hexdigest()
    expected = json.loads(args.python_report.read_text())
    rows, info = corpus(args.data)
    if manifest["training"]["validation_corpus"]["sha256"] != info["sha256"] or expected["development"]["sha256"] != info["sha256"] or expected["test_data_used"]:
        raise ValueError("native replay requires the declared development corpus")
    selection = manifest["spelling"] if "spelling" in manifest else manifest["complete_selection"]
    if selection["selection_sha256"] != hashlib.sha256(args.python_report.read_bytes()).hexdigest():
        raise ValueError("native replay selection differs from the Python report")
    if manifest["vocabulary"] != info["vocabulary"]:
        raise ValueError("native replay vocabulary differs")
    report = {"test_data_used": False, "development": info, "full_vocabulary": info["vocabulary"],
              "selection_sha256": hashlib.sha256(args.python_report.read_bytes()).hexdigest(),
              "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "native_cli_sha256": cli_hash}
    artifacts = {"encoder.bin": manifest["encoder_sha256"], "words.bin": manifest["words_sha256"]}
    if manifest.get("auxiliary_weight", 0.0):
        auxiliary = json.loads((args.model / "auxiliary/manifest.json").read_text())
        artifacts.update({"auxiliary/encoder.bin": auxiliary["encoder_sha256"], "auxiliary/words.bin": auxiliary["words_sha256"]})
    for index, _ in enumerate(manifest.get("additional_models", [])):
        folder = f"additional{index + 1}"
        extra = json.loads((args.model / folder / "manifest.json").read_text())
        artifacts.update({folder + "/encoder.bin": extra["encoder_sha256"], folder + "/words.bin": extra["words_sha256"]})
    if "spelling" in manifest:
        artifacts.update({"spelling.bin": manifest["spelling"]["sha256"],
                          "spelling/encoder.bin": manifest["spelling"]["context_encoder_sha256"]})
    for name, digest in artifacts.items():
        if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"native artifact hash differs: {name}")
    report["native_artifacts_sha256"] = artifacts
    with tempfile.TemporaryDirectory(prefix="drawswipe-native-development-") as directory:
        rebuilt, references = Path(directory) / "rebuilt.bin", Path(directory) / "raw.json"
        subprocess.run([str(args.corpus_cli.resolve()), "--output", str(rebuilt), "--samples", str(len(rows)),
                        "--seed", str(info["seed"]), "--vocab", str(info["vocabulary"]),
                        "--verification", str(references), "--verification-limit", str(len(rows))],
                       check=True, capture_output=True, text=True)
        regenerated_hash = hashlib.sha256(rebuilt.read_bytes()).hexdigest()
        if regenerated_hash != info["sha256"]:
            raise ValueError("regenerated development bytes differ; original data remains unchanged")
        cases = json.loads(references.read_text())
        if len(cases) != len(rows):
            raise ValueError("raw development reference count differs")
        report["regenerated_development_sha256"] = regenerated_hash
        counts = np.zeros((3, 2), dtype=np.int64)
        timings = []
        began = time.monotonic()
        process = subprocess.Popen([str(args.retrieval_cli.resolve()), "--model", str(args.model.resolve())],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            for i, (row, case) in enumerate(zip(rows, cases)):
                # Labels, writer identities, and generation seeds stay outside
                # the inference request. Every row receives the same scorer.
                process.stdin.write(json.dumps({"points": case["points"]}) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f"native replay ended after {i} rows: {process.stderr.read()}")
                result = json.loads(line)
                predictions = result["predictions"]
                if len(predictions) != 5:
                    raise ValueError(f"native replay returned fewer than five predictions at row {i}")
                names = [item["word"] for item in predictions]
                level = int(row["difficulty"])
                counts[level] += [names[0] == case["word"], case["word"] in names]
                timings.append(result["milliseconds"])
                if (i + 1) % 256 == 0 or i + 1 == len(rows):
                    print(json.dumps({"finished": i + 1, "total": len(rows), "seconds": time.monotonic() - began}), flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(process.stderr.read())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        if (args.model / "manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("native manifest changed during development replay")
        if hashlib.sha256(args.retrieval_cli.read_bytes()).hexdigest() != cli_hash:
            raise ValueError("native executable changed during development replay")
        for name, digest in artifacts.items():
            if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"native artifact changed during development replay: {name}")
        metrics, matches = {}, True
        for level, name in enumerate(["clean", "loose", "memory"]):
            size = int((rows["difficulty"] == level).sum())
            metrics[name] = {"samples": size, "top1": int(counts[level, 0]) / size, "top5": int(counts[level, 1]) / size,
                             "correct_top1": int(counts[level, 0]), "correct_top5": int(counts[level, 1])}
            for column, key in enumerate(["top1", "top5"]):
                matches &= int(counts[level, column]) == round(expected["selected"]["validation"][name][key] * size)
        report.update(validation=metrics, matches_python_hit_counts=bool(matches), evaluated_rows=len(cases),
                      native_files_unchanged_through_replay=True,
                      median_milliseconds=float(np.median(timings)), p95_milliseconds=float(np.quantile(timings, 0.95)),
                      limits="Development only; regenerated raw strokes reproduce the unchanged corpus hash. Concurrent training affects timing. No benchmark or blind-confirmation evaluation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not matches:
        raise ValueError("native and Python development hit counts differ; inspect the saved report")


if __name__ == "__main__":
    main()
