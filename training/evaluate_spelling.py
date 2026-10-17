#!/usr/bin/env python3
"""Select spelling scores on the entire declared development corpus."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autoregressive_model import SpellingDecoder, word_tokens
from retrieval_model import build_encoder
from train_autoregressive import evaluate
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    rows, info = corpus(args.data)
    if decoded["metadata"]["encoder_sha256"] != encoder_hash or decoded["words"] != saved["words"]:
        raise ValueError("spelling decoder belongs to a different word encoder")
    for record in [saved, decoded]:
        if record["metadata"]["test_data_used"] or record["metadata"]["validation_corpus"]["sha256"] != info["sha256"]:
            raise ValueError("spelling selection requires the declared development corpus")
    device = "mps"
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    encoder = build_encoder(architecture).to(device).eval().requires_grad_(False)
    encoder.load_state_dict(saved["model"])
    proxies = saved["proxies"].to(device)
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
            similarity, candidates = proxy_scores(embedding, proxies).topk(128, dim=1)
            for name, value in zip(cached, [sequence, raw, embedding, similarity, candidates]):
                cached[name].append(value)
        validation.update({name: torch.cat(values) for name, values in cached.items()})
        chosen, trials = evaluate(model, validation, inputs, targets, lengths)
    recall = {}
    present = (validation["candidates"] == validation["labels"][:, None]).any(1)
    for i, name in enumerate(["clean", "loose", "memory"]):
        recall[name] = float(present[validation["difficulty"] == i].float().mean())
    report = {"encoder_sha256": encoder_hash, "decoder_sha256": decoder_hash,
        "decoder_epoch": decoded["epoch"] + 1, "development": info,
        "test_data_used": False, "full_vocabulary": len(saved["words"]),
        "samples": len(rows), "selected": chosen, "shortlist_recall": recall, "trials": trials,
        "rules": "All development rows; natural top128; exact full words; no target insertion or alias credit"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}), flush=True)


if __name__ == "__main__":
    main()
