#!/usr/bin/env python3
"""Export a trained route encoder and vocabulary for dependency-free Rust inference."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from retrieval_model import GestureEncoder, LAYOUT, curve, features, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return value


def floats(tensor) -> bytes:
    return tensor.detach().cpu().numpy().astype("<f4").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GestureEncoder().eval()
    model.load_state_dict(checkpoint["model"])
    encoded = bytearray(b"SWPRT001")
    # Fold normalization into convolution weights; the native runtime uses only
    # convolution, ReLU, dense layers, and vector normalization.
    for i in range(0, len(model.stages), 3):
        conv, norm = model.stages[i], model.stages[i + 1]
        factor = norm.weight / (norm.running_var + norm.eps).sqrt()
        encoded.extend(floats(conv.weight * factor[:, None, None]))
        encoded.extend(floats((conv.bias - norm.running_mean) * factor + norm.bias))
    for i in [1, 4]:
        encoded.extend(floats(model.head[i].weight))
        encoded.extend(floats(model.head[i].bias))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "encoder.bin").write_bytes(encoded)
    limit = args.vocab or len(checkpoint["words"])
    words, counts, paths = vocabulary(ROOT / "word_freq.txt", limit)
    model = model.to(args.device)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(words), 1024):
            values = torch.tensor(paths[start:start + 1024], device=args.device)
            vectors.append(model(features(values)).cpu().numpy())
    vectors = np.concatenate(vectors)
    bank = bytearray(b"SWPBK001")
    bank.extend(struct.pack("<QII", fingerprint(encoded), len(words), vectors.shape[1]))
    popularity = np.log1p(counts) / np.log1p(counts.max())
    for word, pop, vector in zip(words, popularity, vectors):
        name = word.encode("ascii")
        bank.extend(struct.pack("<H", len(name)))
        bank.extend(name)
        bank.extend(struct.pack("<f", float(pop)))
        bank.extend(vector.astype("<f4").tobytes())
    (args.output / "words.bin").write_bytes(bank)
    verification = []
    with torch.no_grad():
        for word in ["hello", "world", "keyboard", "lunchbox", "alpacas"]:
            raw = np.asarray([LAYOUT[c] for c in word], dtype=np.float64)
            raw[:, 1] *= 1.4
            angle = 0.23
            raw = raw @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            raw = raw * 75 + [420, 190]
            expected = model(features(torch.tensor(curve(raw)[None], device=args.device))).cpu().tolist()[0]
            verification.append({"word": word, "points": raw.tolist(), "embedding": expected})
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    report = {
        "schema_version": 1, "architecture": "conv-48-64-96-128-dense-256-128",
        "points": 64, "channels": 8, "embedding": 128, "vocabulary": len(words),
        "checkpoint_epoch": checkpoint["epoch"] + 1, "validation_score": checkpoint["best_score"],
        "encoder_sha256": hashlib.sha256(encoded).hexdigest(),
        "words_sha256": hashlib.sha256(bank).hexdigest(),
        "training": checkpoint["metadata"], "prior_weight": 0.025,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(words):,} word routes and {len(encoded):,} bytes of encoder weights")


if __name__ == "__main__":
    main()
