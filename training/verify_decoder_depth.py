#!/usr/bin/env python3
"""Verify deeper decoder initialization against an existing native reference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import torch

from autoregressive_model import SpellingDecoder, load_decoder, expand_decoder
from export_retrieval import fingerprint
from export_spelling import spelling_binary
from train_joint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=[6, 9, 12], default=9)
    parser.add_argument("--cli", type=Path, default=Path("../target/release/examples/retrieval_cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, cli = args.model.resolve(), args.cli.resolve()
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    saved, decoder_hash = load_checkpoint(args.decoder)
    if manifest["spelling"]["checkpoint_sha256"] != decoder_hash or "context_model" in saved:
        raise ValueError("depth probe requires the matching frozen-context decoder")
    native = (source / "spelling.bin").read_bytes()
    references = (source / "spelling-verification.json").read_bytes()
    for content, expected in [(native, manifest["spelling"]["sha256"]),
                              (references, manifest["spelling"]["verification_sha256"])]:
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("source spelling reference hash differs")
    tags = struct.unpack_from("<QQQIff", native, 8)
    original = load_decoder(saved["model"], tags[3]).eval()
    if spelling_binary(original, *tags) != native:
        raise ValueError("legacy serialization changed")
    expanded = SpellingDecoder(tags[3], layers=args.layers).eval()
    expand_decoder(expanded, saved["model"])
    weights = spelling_binary(expanded, *tags)
    with tempfile.TemporaryDirectory(prefix="drawswipe-decoder-depth-") as directory:
        fixture = Path(directory) / "model"
        shutil.copytree(source, fixture)
        (fixture / "spelling.bin").write_bytes(weights)
        # This temporary fixture tests initialization parity, not model selection.
        manifest["spelling"].update(sha256=hashlib.sha256(weights).hexdigest(), tag=fingerprint(weights))
        (fixture / "manifest.json").write_text(json.dumps(manifest))
        result = subprocess.run([str(cli), "--model", str(fixture), "--verify-spelling",
                                 str(fixture / "spelling-verification.json")], capture_output=True, text=True, check=True)
        cases = json.loads(references)
        if len(result.stdout.splitlines()) != len(cases):
            raise ValueError("not every initialization reference was verified")
        for case in cases:
            actual = subprocess.run([str(cli), "--model", str(fixture)], input=json.dumps({"points": case["points"]}) + "\n",
                                    capture_output=True, text=True, check=True)
            if [p["word"] for p in json.loads(actual.stdout)["predictions"]] != [p["word"] for p in case["predictions"]]:
                raise ValueError("normal native rankings changed after identity expansion")
    report = {"test_data_used": False, "decoder_checkpoint_sha256": decoder_hash,
              "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "native_cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
              "expanded_weights_sha256": hashlib.sha256(weights).hexdigest(), "layers": args.layers,
              "decoder_parameters": sum(parameter.numel() for parameter in expanded.parameters()),
              "legacy_bytes_unchanged": True, "cases": len(cases), "verification_stdout": result.stdout,
              "limits": "Initialization and serialization parity only. Temporary fixture is removed. No trained gain, development evaluation, benchmark evaluation, or app promotion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
