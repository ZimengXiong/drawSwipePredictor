#!/usr/bin/env python3
"""Export every active component of a fixed development word mixture."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from evaluate_spelling import selected_word_weights
from export_retrieval import fingerprint
from train_joint import load_checkpoint


def auxiliary_components(manifest):
    """Return native component order and bindings, excluding the primary."""
    components = []
    if manifest.get("auxiliary_weight", 0):
        components.append({"directory": "auxiliary", "weight": manifest["auxiliary_weight"],
                           "encoder_tag": manifest["auxiliary_encoder_tag"], "bank_tag": manifest["auxiliary_bank_tag"],
                           "checkpoint_sha256": manifest["auxiliary_checkpoint_sha256"]})
    for index, component in enumerate(manifest.get("additional_models", [])):
        components.append({**component, "directory": f"additional{index + 1}"})
    return components


def word_binding(manifest, bank):
    binding = bytearray(bank)
    for component in auxiliary_components(manifest):
        binding.extend(struct.pack("<QQf", component["encoder_tag"], component["bank_tag"], component["weight"]))
    return fingerprint(binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--component", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.component) <= 3:
        parser.error("supply one to three auxiliary components")
    content = args.selection.read_bytes()
    selection = json.loads(content)
    manifest = json.loads((args.output / "manifest.json").read_text())
    paths = [args.primary, *args.component]
    hashes, words = [], None
    for path in paths:
        saved, source_hash = load_checkpoint(path)
        if words is not None and saved["words"] != words:
            raise ValueError("word mixture vocabularies differ")
        words = saved["words"]
        if saved["metadata"]["test_data_used"] or saved["metadata"]["validation_corpus"]["sha256"] != selection["development"]["sha256"]:
            raise ValueError("word mixture requires declared development writers")
        hashes.append(source_hash)
        del saved
    weights = selected_word_weights(selection, hashes, selection["development"]["sha256"], len(words))
    if manifest["checkpoint_sha256"] != hashes[0] or manifest["vocabulary"] != len(words):
        raise ValueError("export the matching primary model first")
    if manifest["prior_weight"] or manifest.get("tta_degrees", 0) or manifest.get("adaptive_prior", False):
        raise ValueError("word mixtures require base cosine retrieval settings")
    for filename, key in [("encoder.bin", "encoder_sha256"), ("words.bin", "words_sha256")]:
        if hashlib.sha256((args.output / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("primary native artifact hash differs")
    components = []
    for index, (path, source_hash, weight) in enumerate(zip(args.component, hashes[1:], weights[1:])):
        directory = "auxiliary" if index == 0 else f"additional{index}"
        folder = args.output / directory
        subprocess.run([sys.executable, str(Path(__file__).with_name("export_retrieval.py")),
                        "--checkpoint", str(path), "--output", str(folder)], check=True)
        components.append({"weight": weight, "checkpoint_sha256": source_hash,
                           "encoder_tag": fingerprint((folder / "encoder.bin").read_bytes()),
                           "bank_tag": fingerprint((folder / "words.bin").read_bytes())})
    first = components[0]
    manifest.update(auxiliary_weight=first["weight"], auxiliary_encoder_tag=first["encoder_tag"],
                    auxiliary_bank_tag=first["bank_tag"], auxiliary_checkpoint_sha256=first["checkpoint_sha256"],
                    additional_models=components[1:],
                    word_mixture={"selection_sha256": hashlib.sha256(content).hexdigest(),
                                  "components": [{"checkpoint_sha256": key, "weight": weight} for key, weight in zip(hashes, weights)]},
                    complete_selection={"selection_sha256": hashlib.sha256(content).hexdigest(),
                                        "development_sha256": selection["development"]["sha256"],
                                        "proxy_weight": None, "deletion_probability": 0.0,
                                        "validation": selection["selected"]["validation"]})
    for key in ["character", "spelling", "reranker"]:
        manifest.pop(key, None)
    for filename in ["character.bin", "character-verification.json", "spelling.bin", "spelling-verification.json",
                     "reranker.json", "reranker-verification.json"]:
        (args.output / filename).unlink(missing_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"word_models": len(paths), "weights": weights}), flush=True)


if __name__ == "__main__":
    main()
