#!/usr/bin/env python3
"""Mine complete prediction and geometry features from separate writer corpora."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from retrieval_model import curve, vocabulary
from train_proxy import ROOT, corpus

SCHEMA = "complete_geometry_v1"
FEATURES = ["retrieval_cost", "complete_cost", "reserved_zero", "complete_margin", "elastic_cost",
            "point_cost", "start_error", "end_error", "rotation", "smoothing", "length_error",
            "displacement_error", "tangent_error", "unrotated_elastic", "sharp_elastic", "rounded_elastic",
            "minimum_point_cost", "minimum_endpoint_error", "sqrt_retrieval_cost", "sqrt_elastic_cost"]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_inventory(directory):
    return {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in [".bin", ".json"]}


def validate_roles(infos, development, split, samples):
    """Reject held-out training writers and partial or substituted development data."""
    if any(info["vocabulary"] != 60000 or not 1 <= info["samples"] <= 1_000_002 for info in infos):
        raise ValueError("complete features require valid full-vocabulary writer blocks")
    if split == "development":
        if len(infos) != 1 or infos[0]["sha256"] != development["sha256"] or samples != development["samples"]:
            raise ValueError("development must keep every row of the declared corpus")
        return
    if split != "train":
        raise ValueError("unknown feature-data role")
    declared = [development]
    for info in infos:
        if info["seed"] in [101, 102, 9001, 12345, 246810, 700007]:
            raise ValueError("reserved evaluation writer block")
        for other in declared:
            if not (info["writer_end"] <= other["writer_start"] or other["writer_end"] <= info["writer_start"]):
                raise ValueError("training writer ranges overlap each other or development")
        declared.append(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=["train", "development"], required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-development", type=Path)
    parser.add_argument("--seed", type=int, default=900683)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cli", type=Path, default=ROOT / "target/release/examples/retrieval_cli")
    parser.add_argument("--replay-cli", type=Path, default=ROOT / "target/release/examples/replay_memory_writers")
    args = parser.parse_args()
    source_hashes = {name: digest(ROOT / "training" / name)
                     for name in ["prepare_complete_geometry.py", "retrieval_model.py", "train_proxy.py"]}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.samples < 1 or not 1 <= args.workers <= 8 or args.output.suffix != ".npz":
        parser.error("use positive counts, one to eight workers, and an .npz output")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("feature output already exists; use a new artifact name")
    if (args.split == "development") != bool(args.reference_development):
        parser.error("only development requires its existing native accuracy report")
    model = args.model.resolve()
    model_hashes = model_inventory(model)
    manifest = json.loads((model / "manifest.json").read_text())
    if manifest["vocabulary"] != 60000 or "spelling" not in manifest or manifest["prior_weight"]:
        raise ValueError("mine features around a complete full-vocabulary spelling model")
    shards = [corpus(path) for path in args.corpus]
    infos = [info for _, info in shards]
    validate_roles(infos, manifest["training"]["validation_corpus"], args.split, args.samples)
    reference = None
    if args.reference_development:
        reference = json.loads(args.reference_development.read_text())
        if reference["test_data_used"] or reference["evaluated_rows"] != args.samples or reference["full_vocabulary"] != 60000:
            raise ValueError("development reference must cover every row and all candidates")
        if reference["native_manifest_sha256"] != model_hashes["manifest.json"] or reference["development"]["sha256"] != infos[0]["sha256"]:
            raise ValueError("native reference belongs to different model or development bytes")
    edges = np.cumsum([len(rows) for rows, _ in shards])
    if args.samples > edges[-1]:
        raise ValueError("not enough distinct training writers")
    selected = np.arange(args.samples) if args.split == "development" else np.random.default_rng(args.seed).choice(int(edges[-1]), args.samples, replace=False)
    shard_ids = np.searchsorted(edges, selected, side="right")
    labels, writers, difficulties = np.empty(args.samples, np.int64), np.empty(args.samples, np.uint64), np.empty(args.samples, np.int32)
    normalized = np.empty((args.samples, 64, 2), np.float32)
    for index, (rows, _) in enumerate(shards):
        mask = shard_ids == index
        local = selected[mask] - (0 if index == 0 else edges[index - 1])
        labels[mask], writers[mask], difficulties[mask], normalized[mask] = rows["label"][local], rows["writer"][local], rows["difficulty"][local], rows["points"][local]
    words, _, _ = vocabulary(ROOT / "word_freq.txt", 60000)
    word_ids = {word: index for index, word in enumerate(words)}
    word_hash = digest(ROOT / "word_freq.txt")
    features = np.empty((args.samples, 128, len(FEATURES)), np.float32)
    candidates = np.empty((args.samples, 128), np.int32)
    began = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="drawswipe-complete-features-") as directory:
        # Private executable copies allow later builds without changing this job.
        cli, replay = Path(directory) / "predict", Path(directory) / "replay"
        shutil.copy2(args.cli, cli)
        shutil.copy2(args.replay_cli, replay)
        cli_hashes = {"prediction": digest(cli), "raw_replay": digest(replay)}

        def extract(item):
            worker, indices = item
            with tempfile.TemporaryFile(mode="w+") as requests:
                for index in indices:
                    requests.write(json.dumps({"word": words[labels[index]], "writer": int(writers[index]),
                                                "difficulty": ["clean", "loose", "memory"][difficulties[index]]}) + "\n")
                requests.seek(0)
                generator = subprocess.Popen([str(replay)], stdin=requests, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                predictor = subprocess.Popen([str(cli), "--model", str(model), "--complete-features", "--candidates", "128"],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                received = 0
                try:
                    for index, line in zip(indices, generator.stdout):
                        raw = json.loads(line)
                        if set(raw) != {"points"} or np.max(np.abs(curve(raw["points"]) - normalized[index])) > 1e-6:
                            raise ValueError("raw replay does not reproduce the declared normalized stroke")
                        # Ground-truth words, writer identities, and difficulty
                        # remain outside the prediction request.
                        predictor.stdin.write(json.dumps({"points": raw["points"]}) + "\n")
                        predictor.stdin.flush()
                        answer = predictor.stdout.readline()
                        if not answer:
                            raise RuntimeError("native feature process ended: " + predictor.stderr.read())
                        answer = json.loads(answer)
                        result = answer["candidates"]
                        if answer["schema"] != SCHEMA or len(result) != 128:
                            raise ValueError("native feature schema or complete candidate count differs")
                        part = np.array([candidate["features"] for candidate in result], dtype=np.float32)
                        if part.shape != (128, len(FEATURES)) or not np.isfinite(part).all() or np.any(part[:, 2] != 0):
                            raise ValueError("invalid features or unexpected popularity signal")
                        if np.any(np.diff(part[:, 1]) < 0):
                            raise ValueError("complete candidates are not in baseline score order")
                        features[index] = part
                        candidates[index] = [word_ids[candidate["word"]] for candidate in result]
                        received += 1
                        if received % 256 == 0:
                            print(json.dumps({"worker": worker, "finished": received, "total": len(indices), "seconds": time.monotonic() - began}), flush=True)
                    predictor.stdin.close()
                    if generator.wait() or predictor.wait() or received != len(indices):
                        raise RuntimeError("incomplete native feature extraction")
                finally:
                    for process in [generator, predictor]:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(extract, enumerate(np.array_split(np.arange(args.samples), args.workers))))
        if digest(cli) != cli_hashes["prediction"] or digest(replay) != cli_hashes["raw_replay"]:
            raise ValueError("snapshot executables changed during feature extraction")
    if model_inventory(model) != model_hashes or digest(ROOT / "word_freq.txt") != word_hash:
        raise ValueError("source model or dictionary changed during feature extraction")
    baseline = {}
    for level, name in enumerate(["clean", "loose", "memory"]):
        mask = difficulties == level
        counts = {f"correct_top{k}": int((candidates[mask, :k] == labels[mask, None]).any(1).sum()) for k in [1, 5, 128]}
        baseline[name] = {"samples": int(mask.sum()), **counts}
        if reference and any(counts[key] != reference["validation"][name][key] for key in ["correct_top1", "correct_top5"]):
            raise ValueError("feature extraction changed the recorded native development hit counts")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".npz", delete=False) as target:
        temporary = Path(target.name)
        np.savez_compressed(target, features=features, candidates=candidates, labels=labels, difficulty=difficulties)
    temporary.replace(args.output)
    report = {"test_data_used": False, "split": args.split, "samples": args.samples, "full_vocabulary": 60000,
              "feature_schema": SCHEMA, "feature_names": FEATURES, "feature_file_sha256": digest(args.output),
              "source_sha256": source_hashes, "git_commit": git_commit,
              "base_manifest_sha256": model_hashes["manifest.json"], "native_artifacts_sha256": model_hashes,
              "executable_snapshots_sha256": cli_hashes, "corpora": infos, "sampling_seed": args.seed,
              "selected_writers_sha256": hashlib.sha256(writers.tobytes()).hexdigest(),
              "selected_rows_sha256": hashlib.sha256(selected.astype(np.uint64).tobytes()).hexdigest(),
              "dictionary_sha256": word_hash, "baseline": baseline, "seconds": time.monotonic() - began,
              "rules": "Natural complete top128; no target insertion, popularity, or vocabulary-rank features. Raw replay reproduces every cached curve. Labels and difficulty are loss/reporting arrays only.",
              "reference_development_sha256": digest(args.reference_development) if args.reference_development else None}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "baseline": baseline}), flush=True)


if __name__ == "__main__":
    main()
