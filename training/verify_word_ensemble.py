#!/usr/bin/env python3
"""Compare native word-ensemble rankings and costs with Python references."""

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
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("../target/release/examples/retrieval_cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.model / "manifest.json").read_text())
    primary, primary_hash = load_checkpoint(args.primary)
    auxiliary, auxiliary_hash = load_checkpoint(args.auxiliary)
    if manifest["checkpoint_sha256"] != primary_hash or manifest["auxiliary_checkpoint_sha256"] != auxiliary_hash or primary["words"] != auxiliary["words"]:
        raise ValueError("native word-ensemble sources differ")
    if manifest["prior_weight"] or manifest["tta_degrees"] or manifest["adaptive_prior"] or any((args.model / name).exists() for name in ["character.bin", "reranker.json"]):
        raise ValueError("verification requires a pure word ensemble")
    for directory in [args.model, args.model / "auxiliary"]:
        local = json.loads((directory / "manifest.json").read_text())
        for name, key in [("encoder.bin", "encoder_sha256"), ("words.bin", "words_sha256")]:
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != local[key]:
                raise ValueError("native word artifact hash differs")
    models = []
    for saved in [primary, auxiliary]:
        model = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
        model.load_state_dict(saved["model"])
        model.requires_grad_(False)
        models.append((model, saved["proxies"].to("mps")))
    cases = json.loads((args.model / "verification.json").read_text())
    payload = "\n".join(json.dumps({"points": case["points"]}) for case in cases) + "\n"
    result = subprocess.run([str(args.cli.resolve()), "--model", str(args.model.resolve())],
                            input=payload, text=True, capture_output=True, check=True)
    native = [json.loads(line) for line in result.stdout.splitlines()]
    if len(native) != len(cases):
        raise ValueError("native reference row count differs")
    reports = []
    with torch.no_grad():
        for case, output in zip(cases, native):
            paths = torch.tensor(curve(case["points"])[None], device="mps")
            scores = [proxy_scores(model(encoder_features(model, paths)), proxies)[0] for model, proxies in models]
            weight = manifest["auxiliary_weight"]
            costs = (1 - ((1 - weight) * scores[0] + weight * scores[1])).clamp_min(0)
            # Match native alphabetical tie-breaking over the complete bank.
            alphabetical = sorted(range(len(primary["words"])), key=primary["words"].__getitem__)
            indices = torch.tensor(alphabetical, device="mps")
            order = costs[indices].argsort(stable=True)[:5]
            ranked = indices[order].cpu().tolist()
            expected = [primary["words"][i] for i in ranked]
            actual = output["predictions"]
            if [item["word"] for item in actual] != expected:
                raise ValueError(f"native top-five words differ for {case['word']}")
            error = max(abs(item["rating"] - float(costs[i])) for item, i in zip(actual, ranked))
            if error > 0.00001:
                raise ValueError(f"native word costs differ by {error}")
            reports.append({"reference_word": case["word"], "exact_top5_matches": True, "maximum_cost_error": error})
    report = {"primary_sha256": primary_hash, "auxiliary_sha256": auxiliary_hash,
              "auxiliary_weight": manifest["auxiliary_weight"], "full_vocabulary": len(primary["words"]),
              "test_data_used": False, "references": reports,
              "limits": "Transformed canonical reference paths verify implementation parity only; no benchmark accuracy claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
