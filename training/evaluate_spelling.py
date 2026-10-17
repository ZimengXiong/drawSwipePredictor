#!/usr/bin/env python3
"""Select spelling scores on the entire declared development corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from autoregressive_model import SpellingDecoder, word_tokens
from export_ensemble import pair_configuration
from retrieval_model import build_encoder
from train_autoregressive import evaluate
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def retrieval_pair_weight(selection, primary_hash, auxiliary_hash, development_hash, vocabulary):
    """Bind a fixed development selection to every retrieval component."""
    if selection["test_data_used"] or selection["development"]["sha256"] != development_hash:
        raise ValueError("retrieval selection requires the declared development corpus")
    size = selection.get("full_vocabulary", selection["development"].get("vocabulary"))
    if size != vocabulary:
        raise ValueError("retrieval selection vocabulary differs")
    chosen = pair_configuration(selection, primary_hash, auxiliary_hash)
    if chosen["proxy_weight"] is not None:
        raise ValueError("retrieval selection must contain word evidence only")
    return chosen["auxiliary_weight"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path,
                        help="Optional stronger word model for natural candidate retrieval")
    parser.add_argument("--retrieval-auxiliary", type=Path)
    parser.add_argument("--retrieval-selection", type=Path,
                        help="Existing development selection for the two retrieval models")
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.retrieval_auxiliary) != bool(args.retrieval_selection):
        parser.error("auxiliary retrieval requires its saved selection")
    torch.set_num_threads(4)
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    retrieved, retrieval_hash = load_checkpoint(args.retrieval_checkpoint) if args.retrieval_checkpoint else (saved, encoder_hash)
    auxiliary, auxiliary_hash = load_checkpoint(args.retrieval_auxiliary) if args.retrieval_auxiliary else (None, None)
    rows, info = corpus(args.data)
    if decoded["metadata"]["encoder_sha256"] != encoder_hash or decoded["words"] != saved["words"]:
        raise ValueError("spelling decoder belongs to a different word encoder")
    records = [saved, decoded, retrieved] + ([] if auxiliary is None else [auxiliary])
    for record in records:
        if record["words"] != saved["words"]:
            raise ValueError("retrieval vocabulary differs from the spelling vocabulary")
        if record["metadata"]["test_data_used"] or record["metadata"]["validation_corpus"]["sha256"] != info["sha256"]:
            raise ValueError("spelling selection requires the declared development corpus")
    auxiliary_weight, selection_hash = 0.0, None
    if auxiliary is not None:
        content = args.retrieval_selection.read_bytes()
        selection_hash = hashlib.sha256(content).hexdigest()
        auxiliary_weight = retrieval_pair_weight(json.loads(content), retrieval_hash, auxiliary_hash, info["sha256"], len(saved["words"]))
    device = "mps"
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    encoder = build_encoder(architecture).to(device).eval().requires_grad_(False)
    encoder.load_state_dict(saved["model"])
    retrieval_encoder = encoder
    if retrieval_hash != encoder_hash:
        retrieval_architecture = retrieved["metadata"]["arguments"].get("architecture", "small")
        retrieval_encoder = build_encoder(retrieval_architecture).to(device).eval().requires_grad_(False)
        retrieval_encoder.load_state_dict(retrieved["model"])
    proxies = retrieved["proxies"].to(device)
    if auxiliary is not None:
        auxiliary_encoder = build_encoder(auxiliary["metadata"]["arguments"].get("architecture", "small")).to(device).eval().requires_grad_(False)
        auxiliary_encoder.load_state_dict(auxiliary["model"])
        auxiliary_proxies = auxiliary["proxies"].to(device)
    model = SpellingDecoder(getattr(encoder, "input_channels", 8)).to(device).eval()
    model.load_state_dict(decoded["model"])
    model.requires_grad_(False)
    inputs, targets, lengths = [value.to(device) for value in word_tokens(saved["words"])]
    paths = torch.tensor(np.array(rows["points"]), device=device)
    validation = {"labels": torch.tensor(np.array(rows["label"], dtype=np.int64), device=device),
                  "difficulty": torch.tensor(np.array(rows["difficulty"], dtype=np.int64), device=device)}
    cached = {key: [] for key in ["sequence", "raw", "embedding", "similarity", "candidates"]}
    with torch.no_grad():
        for part in paths.split(192):
            sequence, raw, embedding = encode(encoder, part)
            retrieval_embedding = embedding if retrieval_encoder is encoder else encode(retrieval_encoder, part)[2]
            scores = proxy_scores(retrieval_embedding, proxies)
            if auxiliary is not None:
                other = proxy_scores(encode(auxiliary_encoder, part)[2], auxiliary_proxies)
                scores = (1 - auxiliary_weight) * scores + auxiliary_weight * other
            similarity, candidates = scores.topk(128, dim=1)
            for name, value in zip(cached, [sequence, raw, embedding, similarity, candidates]):
                cached[name].append(value)
        validation.update({name: torch.cat(values) for name, values in cached.items()})
        chosen, trials = evaluate(model, validation, inputs, targets, lengths)
    recall = {}
    present = (validation["candidates"] == validation["labels"][:, None]).any(1)
    for i, name in enumerate(["clean", "loose", "memory"]):
        recall[name] = float(present[validation["difficulty"] == i].float().mean())
    report = {"encoder_sha256": encoder_hash, "decoder_sha256": decoder_hash,
        "retrieval_sha256": retrieval_hash,
        "retrieval_auxiliary_sha256": auxiliary_hash, "retrieval_auxiliary_weight": auxiliary_weight,
        "retrieval_selection_sha256": selection_hash,
        "decoder_epoch": decoded["epoch"] + 1, "development": info,
        "test_data_used": False, "full_vocabulary": len(saved["words"]),
        "samples": len(rows), "selected": chosen, "shortlist_recall": recall, "trials": trials,
        "rules": "All development rows; natural top128; exact full words; no target insertion or alias credit"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}), flush=True)


if __name__ == "__main__":
    main()
