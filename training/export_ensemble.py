#!/usr/bin/env python3
"""Export an auxiliary word model selected on development data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from export_retrieval import fingerprint
from train_joint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    auxiliary, source_hash = load_checkpoint(args.auxiliary_checkpoint)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if selection["test_data_used"] or selection["primary_sha256"] != manifest["checkpoint_sha256"] or selection["auxiliary_sha256"] != source_hash:
        raise ValueError("ensemble selection sources differ from the exported models")
    development = selection["development"]["sha256"]
    if development != auxiliary["metadata"]["validation_corpus"]["sha256"] or development != manifest["training"]["validation_corpus"]["sha256"]:
        raise ValueError("ensemble selection requires the declared development corpus")
    if selection["full_vocabulary"] != manifest["vocabulary"] or manifest["vocabulary"] != len(auxiliary["words"]):
        raise ValueError("ensemble vocabularies differ")
    chosen = selection["selected"]
    weight = chosen["auxiliary_weight"]
    if not 0 < weight <= 1:
        raise ValueError("development selected the primary model alone; keep its base export")
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_retrieval.py")),
        "--checkpoint", str(args.auxiliary_checkpoint), "--output", str(args.output / "auxiliary")], check=True)
    encoder = (args.output / "auxiliary/encoder.bin").read_bytes()
    bank = (args.output / "auxiliary/words.bin").read_bytes()
    manifest.update(auxiliary_weight=weight, auxiliary_encoder_tag=fingerprint(encoder), auxiliary_bank_tag=fingerprint(bank),
        auxiliary_checkpoint_sha256=source_hash,
        complete_selection={"development_sha256": development, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
            "proxy_weight": chosen["proxy_weight"], "deletion_probability": chosen["deletion_probability"], "validation": chosen["validation"]})
    # An earlier decoder would not be bound to this complete retrieval context.
    for name in ["character.bin", "character-verification.json"]:
        (args.output / name).unlink(missing_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported auxiliary word evidence with development-selected weight {weight}", flush=True)


if __name__ == "__main__":
    main()
