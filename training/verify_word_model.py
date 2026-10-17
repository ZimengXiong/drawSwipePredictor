#!/usr/bin/env python3
"""Verify full-vocabulary word rankings against one exported native model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from retrieval_model import build_encoder, curve, encoder_features
from train_joint import load_checkpoint
from train_proxy import proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("../target/release/examples/retrieval_cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved, source_hash = load_checkpoint(args.checkpoint)
    manifest_bytes = (args.model / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if saved["metadata"]["test_data_used"]:
        raise ValueError("word parity references must not use test-fitted weights")
    if manifest["checkpoint_sha256"] != source_hash or manifest["vocabulary"] != len(saved["words"]):
        raise ValueError("word verification checkpoint or vocabulary differs")
    if manifest["prior_weight"] or manifest.get("tta_degrees", 0) or manifest.get("adaptive_prior", False) or manifest.get("auxiliary_weight", 0):
        raise ValueError("single word-model verification requires base cosine ranking")
    if any((args.model / name).exists() for name in ["character.bin", "spelling.bin", "reranker.json"]):
        raise ValueError("single word-model verification excludes later rankers")
    artifacts = {}
    for name, key in [("encoder.bin", "encoder_sha256"), ("words.bin", "words_sha256")]:
        digest = hashlib.sha256((args.model / name).read_bytes()).hexdigest()
        if digest != manifest[key]:
            raise ValueError("word artifact hash differs")
        artifacts[name] = digest
    encoder = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval().requires_grad_(False)
    encoder.load_state_dict(saved["model"])
    proxies = saved["proxies"].to("mps")
    cases = json.loads((args.model / "verification.json").read_text())
    expected = []
    with torch.no_grad():
        for case in cases:
            paths = torch.tensor(curve(case["points"])[None], device="mps")
            query = encoder(encoder_features(encoder, paths))
            costs = (1 - proxy_scores(query, proxies)[0]).clamp_min(0).cpu().tolist()
            indices = sorted(range(len(costs)), key=lambda i: (costs[i], saved["words"][i]))[:5]
            expected.append([(saved["words"][i], costs[i]) for i in indices])
    result = subprocess.run([str(args.cli.resolve()), "--model", str(args.model.resolve())],
                            input="".join(json.dumps({"points": case["points"]}) + "\n" for case in cases),
                            capture_output=True, text=True, check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(cases):
        raise ValueError("word verification returned fewer drawings")
    maximum = 0.0
    for output, predictions in zip(outputs, expected):
        actual = output["predictions"]
        if [p["word"] for p in actual] != [word for word, _ in predictions]:
            raise ValueError("native word top-five rankings differ")
        maximum = max(maximum, max(abs(p["rating"] - cost) for p, (_, cost) in zip(actual, predictions)))
    if maximum > 1e-5:
        raise ValueError(f"native word costs differ by {maximum}")
    report = {"test_data_used": False, "checkpoint_sha256": source_hash, "artifacts": artifacts,
              "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "native_cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
              "full_vocabulary": len(saved["words"]), "embedding_dimension": manifest["embedding"],
              "cases": len(cases), "exact_top5": True, "maximum_cost_error": maximum,
              "limits": "Canonical implementation parity only. Normal inference receives points only. This is not development or benchmark accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
