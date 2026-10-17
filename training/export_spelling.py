#!/usr/bin/env python3
"""Export a spelling decoder bound to all selected native word models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import torch

from autoregressive_model import load_decoder, word_tokens
from export_retrieval import fingerprint, floats
from retrieval_model import build_encoder, curve, encoder_features
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import proxy_scores
from export_word_mixture import auxiliary_components, word_binding


def spelling_binary(model, context_tag, encoder_tag, bank_tag, channels, proxy_weight, length_power):
    """Serialize every decoder block while preserving the legacy three-block format."""
    layers = len(model.decoder.layers)
    width = model.symbols.embedding_dim
    if width not in [192, 384] or model.positions.num_embeddings != 64 or not 1 <= layers <= 12:
        raise ValueError("native spelling requires width 192 or 384, 64 token positions, and one to twelve blocks")
    version = b"SWPSP003" if width != 192 else b"SWPSP001" if layers == 3 else b"SWPSP002"
    binary = bytearray(version +
                       struct.pack("<QQQIff", context_tag, encoder_tag, bank_tag, channels, proxy_weight, length_power))
    if version != b"SWPSP001":
        binary.extend(struct.pack("<I", layers))
    if version == b"SWPSP003":
        binary.extend(struct.pack("<I", width))

    def linear(layer):
        binary.extend(floats(layer.weight))
        binary.extend(floats(layer.bias))

    def attention(layer):
        binary.extend(floats(layer.in_proj_weight))
        binary.extend(floats(layer.in_proj_bias))
        linear(layer.out_proj)

    linear(model.memory)
    for tensor in [model.memory_position, model.symbols.weight, model.positions.weight]:
        binary.extend(floats(tensor))
    for block in model.decoder.layers:
        for norm in [block.norm1, block.norm2, block.norm3]:
            linear(norm)
        attention(block.self_attn)
        attention(block.multihead_attn)
        linear(block.linear1)
        linear(block.linear2)
    linear(model.decoder.norm)
    linear(model.output)
    return bytes(binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--auxiliary-checkpoint", type=Path)
    parser.add_argument("--word-checkpoint", type=Path, action="append", default=[],
                        help="Every auxiliary word checkpoint in a complete mixture")
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.auxiliary_checkpoint and args.word_checkpoint:
        parser.error("use legacy auxiliary or complete word checkpoint options")
    torch.set_num_threads(4)
    context, context_hash = load_checkpoint(args.context_checkpoint)
    retrieved, retrieval_hash = load_checkpoint(args.retrieval_checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if selection["test_data_used"] or selection["samples"] != 12000 or selection["full_vocabulary"] != len(retrieved["words"]):
        raise ValueError("spelling export requires full development selection")
    if (selection["encoder_sha256"], selection["decoder_sha256"]) != (context_hash, decoder_hash):
        raise ValueError("spelling selection checkpoint bindings differ")
    if decoded["metadata"]["encoder_sha256"] != context_hash or manifest["checkpoint_sha256"] != retrieval_hash:
        raise ValueError("spelling encoder or native retrieval binding differs")
    components = auxiliary_components(manifest)
    paths = [args.auxiliary_checkpoint] if args.auxiliary_checkpoint else args.word_checkpoint
    loaded = [load_checkpoint(path) for path in paths]
    records_by_hash = {retrieval_hash: retrieved, **{key: record for record, key in loaded}}
    expected = {retrieval_hash: 1 - sum(component["weight"] for component in components),
                **{component["checkpoint_sha256"]: component["weight"] for component in components}}
    if len(expected) != len(components) + 1 or len(records_by_hash) != len(loaded) + 1 or set(records_by_hash) != set(expected):
        raise ValueError("supply every native word component exactly once")
    if components:
        if selection["retrieval_selection_sha256"] != manifest["complete_selection"]["selection_sha256"]:
            raise ValueError("spelling selection uses a different word mixture")
    if selection["retrieval_sha256"] is None:
        selected_components = selection["word_retrieval"]
        selected = {component["sha256"]: component["weight"] for component in selected_components}
        if len(selected) != len(selected_components) or set(selected) != set(expected) or any(not math.isfinite(selected[key]) or abs(selected[key] - weight) > 1e-6 for key, weight in expected.items()):
            raise ValueError("spelling selection must retain every complete word component")
    else:
        if selection["retrieval_sha256"] != retrieval_hash or len(components) > 1:
            raise ValueError("legacy spelling selection cannot discard native word components")
        auxiliary_hash = components[0]["checkpoint_sha256"] if components else None
        auxiliary_weight = components[0]["weight"] if components else 0.0
        if selection.get("retrieval_auxiliary_sha256") != auxiliary_hash or selection.get("retrieval_auxiliary_weight", 0.0) != auxiliary_weight:
            raise ValueError("spelling selection auxiliary context differs")
    records = [context, decoded, *records_by_hash.values()]
    for record in records:
        if record["words"] != retrieved["words"] or record["metadata"]["test_data_used"]:
            raise ValueError("spelling vocabulary or data provenance differs")
        if record["metadata"]["validation_corpus"]["sha256"] != selection["development"]["sha256"]:
            raise ValueError("spelling development corpus differs")
    if "geometry_ranker" in manifest or any((args.output / name).exists() for name in ["reranker.json", "character.bin", "geometry-ranker.json"]):
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
    for component in components:
        tags = [fingerprint((args.output / component["directory"] / name).read_bytes()) for name in ["encoder.bin", "words.bin"]]
        if tags != [component["encoder_tag"], component["bank_tag"]]:
            raise ValueError("native auxiliary artifact binding differs")
    context_directory = args.output / "spelling"
    export_context = args.context_checkpoint
    if "context_model" in decoded:
        context_directory.mkdir(parents=True, exist_ok=True)
        export_context = context_directory / "joint-context.pt"
        torch.save({**context, "model": decoded["context_model"]}, export_context)
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_retrieval.py")),
                    "--checkpoint", str(export_context), "--output", str(context_directory)], check=True)
    context_bytes = (context_directory / "encoder.bin").read_bytes()
    channels = 16 if context["metadata"]["arguments"].get("architecture") in ["intrinsic", "giant", "massive"] else 8
    model = load_decoder(decoded["model"], channels).eval()
    binary = spelling_binary(model, fingerprint(context_bytes), fingerprint(encoder_bytes), word_binding(manifest, bank_bytes),
                             channels, chosen["proxy_weight"], chosen["length_power"])
    device = "mps"
    def encoder(record):
        result = build_encoder(record["metadata"]["arguments"].get("architecture", "small")).to(device).eval()
        result.load_state_dict(record["model"])
        return result
    context_encoder = encoder(context)
    if "context_model" in decoded:
        context_encoder.load_state_dict(decoded["context_model"])
    word_models = [(encoder(records_by_hash[key]), records_by_hash[key]["proxies"].to(device), weight) for key, weight in expected.items()]
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
            scores = sum(weight * proxy_scores(word_encoder(encoder_features(word_encoder, paths)), proxies)
                         for word_encoder, proxies, weight in word_models)
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
        "context_checkpoint_sha256": decoder_hash if "context_model" in decoded else context_hash,
        "context_initializer_sha256": context_hash, "context_encoder_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "joint_context": "context_model" in decoded,
        "decoder_layers": len(model.decoder.layers),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(), "selected": chosen}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(binary):,} bytes of bound spelling weights", flush=True)


if __name__ == "__main__":
    main()
