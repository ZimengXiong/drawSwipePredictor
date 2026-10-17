#!/usr/bin/env python3
"""Verify native spelling parity, artifact hashes, and model binding failures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("../target/release/examples/retrieval_cli"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, cli = args.model.resolve(), args.cli.resolve()
    manifest_bytes = (model / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    spelling = manifest["spelling"]
    artifacts = {"encoder.bin": manifest["encoder_sha256"], "words.bin": manifest["words_sha256"],
                 "spelling.bin": spelling["sha256"], "spelling/encoder.bin": spelling["context_encoder_sha256"],
                 "spelling-verification.json": spelling["verification_sha256"]}
    if manifest.get("auxiliary_weight", 0.0):
        auxiliary = json.loads((model / "auxiliary/manifest.json").read_text())
        artifacts.update({"auxiliary/encoder.bin": auxiliary["encoder_sha256"], "auxiliary/words.bin": auxiliary["words_sha256"]})
    for index, _ in enumerate(manifest.get("additional_models", [])):
        folder = f"additional{index + 1}"
        extra = json.loads((model / folder / "manifest.json").read_text())
        artifacts.update({folder + "/encoder.bin": extra["encoder_sha256"], folder + "/words.bin": extra["words_sha256"]})
    for name, expected in artifacts.items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact hash differs: {name}")
    cases = json.loads((model / "spelling-verification.json").read_text())
    result = subprocess.run([str(cli), "--model", str(model), "--verify-spelling", str(model / "spelling-verification.json")],
                            capture_output=True, text=True, check=True)
    if len(result.stdout.splitlines()) != len(cases):
        raise ValueError("not every reference case was verified")
    failures = {}
    probes = ["missing_decoder", "changed_decoder", "changed_context", "changed_mixture"]
    for index, _ in enumerate(manifest.get("additional_models", [])):
        probes.extend([f"missing_additional{index + 1}", f"changed_additional{index + 1}", f"reweighted_additional{index + 1}"])
    for name in probes:
        with tempfile.TemporaryDirectory(prefix="drawswipe-spelling-binding-") as directory:
            copied = Path(directory) / "model"
            shutil.copytree(model, copied)
            if name == "missing_decoder":
                (copied / "spelling.bin").unlink()
            elif name in ["changed_decoder", "changed_context"]:
                path = copied / ("spelling.bin" if name == "changed_decoder" else "spelling/encoder.bin")
                content = bytearray(path.read_bytes())
                content[44 if name == "changed_decoder" else 8] ^= 1
                path.write_bytes(content)
            elif name.startswith("missing_additional"):
                (copied / name.removeprefix("missing_") / "encoder.bin").unlink()
            elif name.startswith("changed_additional"):
                path = copied / name.removeprefix("changed_") / "words.bin"
                content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
            else:
                path = copied / "manifest.json"
                modified = json.loads(path.read_text())
                if name.startswith("reweighted_additional"):
                    index = int(name.removeprefix("reweighted_additional")) - 1
                    modified["additional_models"][index]["weight"] *= 0.5
                else:
                    modified["auxiliary_weight"] = 0.5 if manifest.get("auxiliary_weight") != 0.5 else 0.25
                path.write_text(json.dumps(modified))
            failed = subprocess.run([str(cli), "--model", str(copied)], input="", capture_output=True, text=True)
            if failed.returncode == 0:
                raise ValueError(f"native inference accepted {name}")
            failures[name] = {"exit_code": failed.returncode, "reason": failed.stderr.strip()}
    timings = []
    for case in cases:
        # Normal inference receives points only. Expected words stay in this verifier.
        output = subprocess.run([str(cli), "--model", str(model)], input=json.dumps({"points": case["points"]}) + "\n",
                                capture_output=True, text=True, check=True)
        actual = json.loads(output.stdout)
        if [p["word"] for p in actual["predictions"]] != [p["word"] for p in case["predictions"]]:
            raise ValueError("normal CLI rankings differ from the verified rankings")
        timings.append(actual["milliseconds"])
    report = {"test_data_used": False, "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "native_cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(), "artifacts": artifacts,
              "cases": len(cases), "verification_stdout": result.stdout, "binding_rejections": failures,
              "normal_inference_milliseconds": timings,
              "limits": "Canonical implementation parity only; this is not development or benchmark accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
