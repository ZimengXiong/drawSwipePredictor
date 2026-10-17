#!/usr/bin/env python3
"""Export a spelling decoder bound to all selected native word models."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import torch

from autoregressive_model import SpellingDecoder, word_tokens
from export_retrieval import fingerprint, floats
from retrieval_model import build_encoder, curve, encoder_features
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--auxiliary-checkpoint", type=Path)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    context, context_hash = load_checkpoint(args.context_checkpoint)
    retrieved, retrieval_hash = load_checkpoint(args.retrieval_checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if selection["test_data_used"] or selection["samples"] != 12000 or selection["full_vocabulary"] != len(retrieved["words"]):
        raise ValueError("spelling export requires full development selection")
    if (selection["encoder_sha256"], selection["decoder_sha256"], selection["retrieval_sha256"]) != (context_hash, decoder_hash, retrieval_hash):
        raise ValueError("spelling selection checkpoint bindings differ")
    if decoded["metadata"]["encoder_sha256"] != context_hash or manifest["checkpoint_sha256"] != retrieval_hash:
        raise ValueError("spelling encoder or native retrieval binding differs")
    auxiliary_weight = manifest.get("auxiliary_weight", 0.0)
    auxiliary = None
    auxiliary_hash = None
    records = [context, retrieved, decoded]
    if auxiliary_weight:
        if not args.auxiliary_checkpoint:
            raise ValueError("selected retrieval requires its auxiliary checkpoint")
        auxiliary, auxiliary_hash = load_checkpoint(args.auxiliary_checkpoint)
        records.append(auxiliary)
        if selection["retrieval_selection_sha256"] != manifest["complete_selection"]["selection_sha256"]:
            raise ValueError("spelling selection uses a different word mixture")
        if auxiliary_hash != manifest["auxiliary_checkpoint_sha256"]:
            raise ValueError("native auxiliary checkpoint differs")
    if selection.get("retrieval_auxiliary_sha256") != auxiliary_hash or selection.get("retrieval_auxiliary_weight", 0.0) != auxiliary_weight:
        raise ValueError("spelling selection auxiliary context differs")
    for record in records:
        if record["words"] != retrieved["words"] or record["metadata"]["test_data_used"]:
            raise ValueError("spelling vocabulary or data provenance differs")
        if record["metadata"]["validation_corpus"]["sha256"] != selection["development"]["sha256"]:
            raise ValueError("spelling development corpus differs")
    if any((args.output / name).exists() for name in ["reranker.json", "character.bin"]):
        raise ValueError("export spelling into a word-only model directory")
    if manifest["prior_weight"] or manifest.get("tta_degrees", 0) or manifest.get("adaptive_prior", False):
        raise ValueError("spelling requires the selected base retrieval settings")
    chosen = selection["selected"]
    if chosen["proxy_weight"] is None:
        raise ValueError("selection chose word retrieval alone")
    encoder_bytes = (args.output / "encoder.bin").read_bytes()
    bank_bytes = (args.output / "words.bin").read_bytes()
    for name, content in [("encoder", encoder_bytes), ("words", bank_bytes)]:
        if hashlib.sha256(content).hexdigest() != manifest[name + "_sha256"]:
            raise ValueError("native retrieval artifact hash differs")
    binding = bytearray(bank_bytes)
    if auxiliary_weight:
        tags = [fingerprint((args.output / "auxiliary" / name).read_bytes()) for name in ["encoder.bin", "words.bin"]]
        if tags != [manifest["auxiliary_encoder_tag"], manifest["auxiliary_bank_tag"]]:
            raise ValueError("native auxiliary artifact binding differs")
        binding.extend(struct.pack("<QQf", *tags, auxiliary_weight))
    context_directory = args.output / "spelling"
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_retrieval.py")),
                    "--checkpoint", str(args.context_checkpoint), "--output", str(context_directory)], check=True)
    context_bytes = (context_directory / "encoder.bin").read_bytes()
    channels = 16 if context["metadata"]["arguments"].get("architecture") in ["intrinsic", "giant"] else 8
    model = SpellingDecoder(channels).eval()
    model.load_state_dict(decoded["model"])
    binary = bytearray(b"SWPSP001" + struct.pack("<QQQIff", fingerprint(context_bytes), fingerprint(encoder_bytes),
                                              fingerprint(binding), channels, chosen["proxy_weight"], chosen["length_power"]))
    def linear(layer):
        binary.extend(floats(layer.weight)); binary.extend(floats(layer.bias))
    def attention(layer):
        binary.extend(floats(layer.in_proj_weight)); binary.extend(floats(layer.in_proj_bias))
        linear(layer.out_proj)
    linear(model.memory)
    for tensor in [model.memory_position, model.symbols.weight, model.positions.weight]:
        binary.extend(floats(tensor))
    for block in model.decoder.layers:
        for norm in [block.norm1, block.norm2, block.norm3]:
            linear(norm)
        attention(block.self_attn); attention(block.multihead_attn)
        linear(block.linear1); linear(block.linear2)
    linear(model.decoder.norm); linear(model.output)
    device = "mps"
    def encoder(record):
        result = build_encoder(record["metadata"]["arguments"].get("architecture", "small")).to(device).eval()
        result.load_state_dict(record["model"])
        return result
    context_encoder, retrieval_encoder = encoder(context), encoder(retrieved)
    auxiliary_encoder = encoder(auxiliary) if auxiliary else None
    proxies = retrieved["proxies"].to(device)
    auxiliary_proxies = auxiliary["proxies"].to(device) if auxiliary else None
    model = model.to(device)
    words = retrieved["words"]
    inputs, targets, lengths = [v.to(device) for v in word_tokens(words)]
    cases = json.loads((args.output / "verification.json").read_text())
    verification = []
    with torch.no_grad():
        for case in cases:
            paths = torch.tensor(curve(case["points"])[None], device=device)
            sequence, raw, embedding = encode(context_encoder, paths)
            memory = model.encode_memory(sequence, raw, embedding)
            scores = proxy_scores(retrieval_encoder(encoder_features(retrieval_encoder, paths)), proxies)
            if auxiliary_encoder:
                other = proxy_scores(auxiliary_encoder(encoder_features(auxiliary_encoder, paths)), auxiliary_proxies)
                scores = (1 - auxiliary_weight) * scores + auxiliary_weight * other
            similarity, candidates = scores.topk(128, dim=1)
            costs = model.candidate_cost(memory, candidates, inputs, targets)
            ratings = costs / lengths[candidates].float().pow(chosen["length_power"]) - 32 * chosen["proxy_weight"] * similarity
            ids = candidates[0].cpu().tolist()
            values = ratings[0].cpu().tolist()
            order = sorted(range(len(ids)), key=lambda i: (values[i], words[ids[i]]))[:5]
            predictions = [{"word": words[ids[i]], "rating": values[i]} for i in order]
            probes = list(dict.fromkeys([words[i] for i in ids] + ["bookkeeper", "mississippi", "letter", "hello"]))
            probe_inputs, probe_targets, _ = [v.to(device) for v in word_tokens(probes)]
            probe_ids = torch.arange(len(probes), device=device)[None]
            probe_costs = model.candidate_cost(memory, probe_ids, probe_inputs, probe_targets)[0].cpu().tolist()
            verification.append({"word": case["word"], "points": case["points"], "words": probes,
                                 "costs": probe_costs, "predictions": predictions})
    for name, content in {"spelling.bin": bytes(binary), "spelling-verification.json": (json.dumps(verification, indent=2) + "\n").encode()}.items():
        temporary = args.output / (name + ".tmp")
        temporary.write_bytes(content); temporary.replace(args.output / name)
    manifest["spelling"] = {"sha256": hashlib.sha256(binary).hexdigest(), "tag": fingerprint(binary), "checkpoint_sha256": decoder_hash,
        "verification_sha256": hashlib.sha256((args.output / "spelling-verification.json").read_bytes()).hexdigest(),
        "context_checkpoint_sha256": context_hash, "context_encoder_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(), "selected": chosen}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(binary):,} bytes of bound spelling weights", flush=True)


if __name__ == "__main__":
    main()
