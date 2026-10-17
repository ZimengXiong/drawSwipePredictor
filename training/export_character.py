#!/usr/bin/env python3
"""Export a bound native character decoder and CLI parity reference cases."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

import torch

from character_edit import ctc_edit_components, edit_log_likelihood
from character_model import CharacterDecoder, character_targets, candidate_ctc_cost
from export_retrieval import fingerprint, floats
from retrieval_model import build_encoder, curve, encoder_features
from train_character import encode
from train_proxy import proxy_scores


def read_checkpoint(path):
    content = path.read_bytes()
    return torch.load(io.BytesIO(content), map_location="cpu", weights_only=False), hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--auxiliary-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    retriever, retriever_hash = read_checkpoint(args.retrieval_checkpoint)
    decoder, decoder_hash = read_checkpoint(args.decoder_checkpoint or args.retrieval_checkpoint)
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest.get("checkpoint_sha256") != retriever_hash:
        raise ValueError("export the matching retrieval checkpoint first")
    if "character_model" not in decoder and decoder["metadata"]["encoder_sha256"] != retriever_hash:
        raise ValueError("character decoder belongs to a different frozen encoder")
    if "character_model" in decoder and decoder_hash != retriever_hash:
        raise ValueError("joint decoder and encoder must come from the same checkpoint")
    selected = manifest.get("complete_selection", decoder["selected"])
    weight = selected["proxy_weight"]
    if weight is None:
        raise ValueError("development selected retrieval alone; use its base export")
    probability = selected.get("deletion_probability", 0.0)
    architecture = retriever["metadata"]["arguments"].get("architecture", "small")
    encoder = build_encoder(architecture).to("mps").eval()
    encoder.load_state_dict(retriever["model"])
    channels = getattr(encoder, "input_channels", 8)
    model = CharacterDecoder(channels).to("mps").eval()
    model.load_state_dict(decoder.get("character_model", decoder["model"]))
    encoder_bytes, bank_bytes = (args.output / "encoder.bin").read_bytes(), (args.output / "words.bin").read_bytes()
    for key, content in [("encoder", encoder_bytes), ("words", bank_bytes)]:
        if manifest[key + "_sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError("native base artifact hash mismatch")
    binding = bytearray(bank_bytes)
    auxiliary_model = None
    auxiliary_weight = manifest.get("auxiliary_weight", 0.0)
    if auxiliary_weight:
        if not args.auxiliary_checkpoint or "complete_selection" not in manifest:
            raise ValueError("ensemble character decoding requires its auxiliary checkpoint and development selection")
        auxiliary, auxiliary_hash = read_checkpoint(args.auxiliary_checkpoint)
        if auxiliary_hash != manifest["auxiliary_checkpoint_sha256"] or auxiliary["words"] != retriever["words"]:
            raise ValueError("auxiliary character evidence source differs")
        auxiliary_model = build_encoder(auxiliary["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
        auxiliary_model.load_state_dict(auxiliary["model"])
        auxiliary_proxies = auxiliary["proxies"].to("mps")
        binding.extend(struct.pack("<QQf", manifest["auxiliary_encoder_tag"], manifest["auxiliary_bank_tag"], auxiliary_weight))
    binary = bytearray(b"SWPCH001" + struct.pack("<QQIfff", fingerprint(encoder_bytes), fingerprint(binding), channels, weight, probability, 0.3))
    def convolution(conv, norm=None):
        if norm is None:
            binary.extend(floats(conv.weight)); binary.extend(floats(conv.bias))
        else:
            factor = norm.weight / (norm.running_var + norm.eps).sqrt()
            binary.extend(floats(conv.weight * factor[:, None, None]))
            binary.extend(floats((conv.bias - norm.running_mean) * factor + norm.bias))
    convolution(model.stem[0], model.stem[1])
    for block in model.blocks:
        convolution(block.layers[0], block.layers[1]); convolution(block.layers[3], block.layers[4])
    convolution(model.output)
    targets, lengths, _ = character_targets(retriever["words"])
    proxies = retriever["proxies"].to("mps")
    cases = json.loads((args.output / "verification.json").read_text())
    verification = []
    with torch.no_grad():
        for case in cases:
            paths = torch.tensor(curve(case["points"])[None], device="mps")
            sequence, raw, embedding = encode(encoder, paths)
            log_probs = model(sequence, raw, embedding).log_softmax(2)
            scores = proxy_scores(embedding, proxies)
            if auxiliary_model is not None:
                other = proxy_scores(auxiliary_model(encoder_features(auxiliary_model, paths)), auxiliary_proxies)
                scores = (1 - auxiliary_weight) * scores + auxiliary_weight * other
            similarity, candidates = scores.topk(128, dim=1)
            sizes = lengths[candidates.cpu()].to("mps")
            if probability:
                ordinary, deleted = ctc_edit_components(log_probs, candidates, targets, lengths)
                likelihood = edit_log_likelihood(ordinary, deleted, sizes, probability)
            else:
                likelihood = -candidate_ctc_cost(log_probs, candidates, targets, lengths)
            costs = -likelihood / sizes.float().pow(0.3) - weight * 32 * similarity
            order = costs.argsort(dim=1, stable=True)[0, :5]
            predictions = [{"word": retriever["words"][int(candidates[0, j])], "rating": float(costs[0, j])} for j in order]
            verification.append({"word": case["word"], "points": case["points"], "log_probabilities": log_probs[0].cpu().tolist(), "predictions": predictions})
    for name, content in {"character.bin": bytes(binary), "character-verification.json": (json.dumps(verification, indent=2) + "\n").encode()}.items():
        temporary = args.output / (name + ".tmp")
        temporary.write_bytes(content); temporary.replace(args.output / name)
    manifest["character"] = {"sha256": hashlib.sha256(binary).hexdigest(), "checkpoint_sha256": decoder_hash,
        "proxy_weight": weight, "deletion_probability": probability, "length_power": 0.3,
        "selection": "Development only; exact-word scoring and naturally retrieved top128 candidates"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(binary):,} bytes of bound character weights", flush=True)


if __name__ == "__main__":
    main()
