#!/usr/bin/env python3
"""Build a macOS app that opens the memory recognizer without a terminal."""

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def model_files(directory: Path) -> list[str]:
    """Include every native inference component and verify declared hashes."""
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {"encoder.bin": manifest["encoder_sha256"], "words.bin": manifest["words_sha256"]}
    files = ["manifest.json", *expected]
    if manifest.get("auxiliary_weight", 0):
        files.extend(["auxiliary/encoder.bin", "auxiliary/words.bin"])
        auxiliary_manifest = directory / "auxiliary/manifest.json"
        if auxiliary_manifest.exists():
            auxiliary = json.loads(auxiliary_manifest.read_text())
            files.append("auxiliary/manifest.json")
            expected.update({"auxiliary/encoder.bin": auxiliary["encoder_sha256"], "auxiliary/words.bin": auxiliary["words_sha256"]})
    for index, _ in enumerate(manifest.get("additional_models", [])):
        folder = f"additional{index + 1}"
        extra = json.loads((directory / folder / "manifest.json").read_text())
        files.extend([folder + "/manifest.json", folder + "/encoder.bin", folder + "/words.bin"])
        expected.update({folder + "/encoder.bin": extra["encoder_sha256"], folder + "/words.bin": extra["words_sha256"]})
    for component, filename in [("character", "character.bin"), ("reranker", "reranker.json"), ("spelling", "spelling.bin"), ("geometry_ranker", "geometry-ranker.json")]:
        if (directory / filename).exists() or component in manifest:
            files.append(filename)
            if component in manifest:
                expected[filename] = manifest[component]["sha256"]
    if "spelling" in manifest:
        files.append("spelling/encoder.bin")
        expected["spelling/encoder.bin"] = manifest["spelling"]["context_encoder_sha256"]
    for filename in ["verification.json", "character-verification.json", "reranker-verification.json", "spelling-verification.json"]:
        if (directory / filename).exists():
            files.append(filename)
    for filename in files:
        path = directory / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename in expected and actual != expected[filename]:
            raise ValueError(f"model artifact hash differs: {filename}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "crates/swipe-engine/models/retrieval")
    parser.add_argument("--output", type=Path, default=ROOT / "build/Draw from Memory.app")
    parser.add_argument("--bundle-id", default="com.zimeng.draw-swipe-memory")
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("For other platforms, run cargo run --release")
    source, output = args.model.resolve(), args.output.resolve()
    if output.suffix != ".app":
        parser.error("output must end with .app")
    files = model_files(source)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "swipe-desktop", "--no-default-features", "--bin", "swipe-learned"],
        cwd=ROOT, check=True,
    )
    subprocess.run(["cargo", "build", "--release", "-p", "swipe-engine", "--example", "retrieval_cli"], cwd=ROOT, check=True)
    cli = ROOT / "target/release/examples/retrieval_cli"
    subprocess.run([str(cli), "--model", str(source)], input="", text=True, check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".drawswipe-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        contents = staged / "Contents"
        executable_dir = contents / "MacOS"
        executable_dir.mkdir(parents=True)
        shutil.copy2(ROOT / "target/release/swipe-learned", executable_dir / "swipe-memory")
        model = contents / "Resources/model"
        for filename in files:
            destination = model / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / filename, destination)
        if model_files(model) != files:
            raise ValueError("packaged model inventory differs")
        subprocess.run([str(cli), "--model", str(model)], input="", text=True, check=True)
        receipt = {"source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                   "artifacts": {filename: hashlib.sha256((model / filename).read_bytes()).hexdigest() for filename in files}}
        (contents / "Resources/model-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (contents / "Info.plist").open("wb") as target:
            plistlib.dump({
                "CFBundleExecutable": "swipe-memory",
                "CFBundleIdentifier": args.bundle_id,
                "CFBundleName": output.stem,
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "0.1.0",
                "NSHighResolutionCapable": True,
            }, target)
        previous = Path(temporary) / "previous.app"
        if output.exists():
            output.rename(previous)
        try:
            staged.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise
    print(output)


if __name__ == "__main__":
    main()
